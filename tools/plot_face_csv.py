"""Plot the face-perception telemetry written by `exploration/face_probe.py --csv`.

The CSV exists to answer one question before any of this is allowed near a
servo: do the derived channels carry usable signal, or just noise? Reading it as
a spreadsheet will not tell you. These are the four views that will:

  1. Detection + IPD     — whether a 3-D fix existed, and whether it was worth
                           anything. "One eye only" is a distinct failure from
                           "not seen" (a camera occluded vs out of range).
                           Interpupillary distance is fixed on a real face, so
                           every wobble in it is measurement error in the same
                           units as the head position below: a trace wandering
                           by 10 mm means the position is worth about that too.
  2. Head position       — X/Y/Z and distance, with approach speed alongside.
  3. Head pose           — the raw yaw/pitch attention is thresholded from,
                           with the cone drawn, plus detected nod/shake spans
                           in pink. Also read it for boundary chatter: the
                           attending ribbon flickering while yaw sits near the
                           threshold is a threshold problem, not a person
                           moving, and wants hysteresis.
  4. Affect + state      — arousal, valence, change rate, shaded by the derived
                           5-state classification (see classify_affect). What
                           you are looking for is whether arousal and valence
                           MOVE when the person's expression moves and hold
                           still when it does not — and whether the state
                           shading changes at moments you actually remember.

Run:
  python tools/plot_face_csv.py exploration/out/face.csv
  python tools/plot_face_csv.py exploration/out/face.csv --save face.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LEGEND_KW = dict(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1,
                 borderaxespad=0.0, frameon=False)
# Panels carrying a twinx need the legend pushed clear of the right-hand tick
# labels, which occupy the strip the default anchor sits in.
LEGEND_TWIN = dict(LEGEND_KW, bbox_to_anchor=(1.08, 1.0))

IPD_LO, IPD_HI = 0.055, 0.072      # plausible adult interpupillary distance (m)

# Must match ATTEND_YAW_DEG / ATTEND_PITCH_DEG in exploration/face_probe.py.
# Kept as literals rather than imported: face_probe pulls in mediapipe and the
# whole perception stack, which is a lot to load to draw two dashed lines.
ATTEND_YAW_DEG, ATTEND_PITCH_DEG = 25.0, 20.0


def load(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"{path} is empty")

    def col(key):
        out = []
        for r in rows:
            v = r.get(key, "")
            out.append(np.nan if v in ("", None) else float(v))
        return np.array(out, dtype=float)

    d = {k: col(k) for k in
         ("t", "det_l", "det_r", "dist", "ipd", "yaw", "pitch", "roll",
          "arousal", "valence", "change", "dwell", "absent", "approach",
          "attending", "det_ms")}
    d["pos"] = np.stack([col("x"), col("y"), col("z")], axis=1)
    return d


# Vertical extent of the nod/shake shading, in degrees. Deliberately not the
# full axis height: the band marks WHEN a gesture happened without hiding the
# traces that show it happening.
GESTURE_BAND = (-20.0, 40.0)


GESTURE_HZ = (0.8, 4.0)     # a head nod/shake, in cycles per second
MIN_EVENT_S = 0.30          # shorter than this is not a gesture
MERGE_GAP_S = 0.30          # detections closer than this are one event


def _half_cycles(w, hyst):
    """Alternations of a mean-centred window, with hysteresis.

    NOT zero crossings. A noisy signal sitting near its own mean crosses zero
    constantly -- counting those reported 31 shakes in a take containing one.
    A Schmitt trigger only alternates once the signal has actually travelled
    past +/-hyst, so noise has to be as large as the gesture to fool it.
    """
    state, count = 0, 0
    for v in w:
        if v > hyst and state <= 0:
            state, count = 1, count + (state != 0)
        elif v < -hyst and state >= 0:
            state, count = -1, count + (state != 0)
    return count


def _noise_sigma(sig):
    """Robust per-signal noise floor, in degrees.

    Taken from successive differences rather than the spread of the signal
    itself, so a take containing real head movement does not inflate its own
    noise estimate. Scaled from the median absolute increment, which for white
    noise sits at about 1.35 sigma.
    """
    v = sig[np.isfinite(sig)]
    if v.size < 3:
        return 0.0
    d = np.abs(np.diff(v))
    return float(np.median(d) / 1.35) if d.size else 0.0


def _clean(t, mask, min_dur=MIN_EVENT_S, gap=MERGE_GAP_S):
    """Merge near-adjacent detections, then drop the ones too short to be real."""
    spans = _spans(t, mask)
    if not spans:
        return mask
    merged = [list(spans[0])]
    for x0, x1 in spans[1:]:
        if x0 - merged[-1][1] <= gap:
            merged[-1][1] = x1
        else:
            merged.append([x0, x1])
    out = np.zeros_like(mask, dtype=bool)
    for x0, x1 in merged:
        if x1 - x0 >= min_dur:
            out |= (t >= x0) & (t <= x1)
    return out


def detect_head_gestures(t, yaw, pitch, win=1.0, min_amp=8.0):
    """Mark frames inside a nod (pitch oscillating) or a shake (yaw).

    Deliberately the simplest thing that can work, on signals already in the
    CSV -- no new inference. Inside a sliding window: subtract the mean, so a
    head held off-centre is not mistaken for motion; require peak-to-peak swing
    above `min_amp`, so jitter is not either; then count hysteretic alternations
    and check the implied rate falls in GESTURE_HZ. That last test is what
    separates a shake from someone slowly turning to look at something -- both
    swing far, only one swings repeatedly.

    A nod and a shake cannot both be scored on one frame: real head gestures
    leak into the other axis, so the larger swing wins.
    """
    n = len(t)
    amp = {"nod": np.zeros(n), "shake": np.zeros(n)}
    # Both bars are set off the signal's own noise floor. A fixed threshold is
    # either deaf on a clean rig or, as measured here, permanently triggered on
    # a noisy one -- head-pose noise on a marginal detection is several degrees,
    # which is a good fraction of a real gesture.
    sigma = {"nod": _noise_sigma(pitch), "shake": _noise_sigma(yaw)}
    floor = {k: max(min_amp, 6.0 * s) for k, s in sigma.items()}

    for i in range(n):
        lo = np.searchsorted(t, t[i] - win / 2.0, "left")
        hi = np.searchsorted(t, t[i] + win / 2.0, "right")
        if hi - lo < 5:
            continue
        span = float(t[hi - 1] - t[lo]) or win
        for key, sig in (("nod", pitch), ("shake", yaw)):
            w = sig[lo:hi]
            if not np.all(np.isfinite(w)):      # a dropout breaks the window
                continue
            w = w - w.mean()
            ptp = float(np.ptp(w))
            if ptp < floor[key]:
                continue
            halves = _half_cycles(w, max(0.30 * ptp, 3.0 * sigma[key], 2.0))
            if halves < 3:                      # under 1.5 swings is not a gesture
                continue
            if GESTURE_HZ[0] <= (halves / 2.0) / span <= GESTURE_HZ[1]:
                amp[key][i] = ptp

    nod = _clean(t, (amp["nod"] > 0) & (amp["nod"] >= amp["shake"]))
    shake = _clean(t, (amp["shake"] > 0) & (amp["shake"] > amp["nod"]))
    return nod, shake


# The 2x2 affect grid plus a neutral centre. Colour carries meaning here:
# warm = high arousal, cool = low, saturated = strong valence either way.
STATE_NAMES = ["unknown", "neutral", "content", "excited", "sad", "angry"]
STATE_COLORS = {
    # 'unknown' is not a sixth quadrant, it is the absence of an observation:
    # no face, so nothing to classify. Kept distinct from neutral because a
    # rule reading "Neutral == 1" must not fire when someone left the room.
    "unknown": "0.85",
    "neutral": "0.62",
    "content": "tab:green",       # low arousal, positive valence
    "excited": "gold",            # high arousal, positive valence
    "sad": "tab:blue",            # low arousal, negative valence
    "angry": "tab:red",           # high arousal, negative valence
}


def classify_affect(arousal, valence, a_thresh=0.0, v_thresh=0.0,
                    deadband=0.20, hyst=0.65, have_face=None):
    """Label each frame with an affect state, or 'unknown' where no face.

    Two independent sign tests, NOT a weighted sum of the two axes. Any single
    weighted combination projects the plane onto a line, which collides the
    diagonally opposite quadrants: with equal weights, angry (+arousal,
    -valence) and content (-arousal, +valence) both score zero.

    Thresholds are applied to BASELINE-CORRECTED values. A resting face is not
    (0, 0) -- resting brow position varies between people, and arousal here is
    built from brow raise, eye widening and jaw open, all of which are
    additions to a neutral face, so its resting value sits near the bottom of
    its range rather than in the middle. The baseline is the take's own median.

    `neutral` is a radius test, not a fifth quadrant: close enough to baseline
    that the quadrant is noise. It carries hysteresis (leave at `deadband`,
    return at `deadband*hyst`) because without it a near-neutral face rattles
    between all four states, which is the circular version of the boundary
    chatter the attention threshold already showed.

    `have_face` marks frames with no detection. Those are 'unknown', not
    'neutral' -- neutral is a claim about a face that was looked at, and
    conflating the two makes an absence indistinguishable from a calm person.
    Baselines are computed only over frames that HAVE a face, so a take that is
    mostly empty room does not drag the median toward whatever the affect
    signals read as when there is nothing to read.
    """
    if have_face is None:
        have_face = np.ones(len(arousal), bool)
    have_face = np.asarray(have_face, bool)
    seen = have_face & np.isfinite(arousal) & np.isfinite(valence)
    a_base = float(np.median(arousal[seen])) if seen.any() else 0.0
    v_base = float(np.median(valence[seen])) if seen.any() else 0.0
    da, dv = arousal - a_base, valence - v_base
    radius = np.hypot(da, dv)

    out = np.full(len(arousal), "neutral", dtype=object)
    in_neutral = True
    for i in range(len(arousal)):
        if not have_face[i] or not (np.isfinite(da[i]) and np.isfinite(dv[i])):
            out[i], in_neutral = "unknown", True
            continue
        # Harder to leave a state than to stay in it: hold neutral until the
        # radius clears `deadband`, then hold the quadrant until it falls back
        # below `deadband*hyst`. Inverting these two makes the deadband easier
        # to escape than to re-enter, which produces MORE chatter, not less.
        bar = deadband if in_neutral else deadband * hyst
        if radius[i] < bar:
            out[i], in_neutral = "neutral", True
            continue
        in_neutral = False
        hi_a, pos_v = da[i] > a_thresh, dv[i] > v_thresh
        out[i] = {(False, True): "content", (True, True): "excited",
                  (False, False): "sad", (True, False): "angry"}[(hi_a, pos_v)]
    return out, (a_base, v_base)


def _spans(t, mask):
    """Contiguous [t0, t1] runs where mask is True -- for shading state ribbons."""
    out, start = [], None
    for i, on in enumerate(mask):
        if on and start is None:
            start = t[i]
        elif not on and start is not None:
            out.append((start, t[i]))
            start = None
    if start is not None:
        out.append((start, t[-1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--save", type=Path, default=None,
                    help="write PNG instead of showing")
    ap.add_argument("--gesture-window", type=float, default=1.0,
                    help="sliding window (s) for nod/shake detection")
    ap.add_argument("--gesture-amp", type=float, default=8.0,
                    help="minimum peak-to-peak swing (deg) to count as a gesture")
    ap.add_argument("--a-thresh", type=float, default=0.0,
                    help="high/low arousal split, RELATIVE to the take's median")
    ap.add_argument("--v-thresh", type=float, default=0.0,
                    help="positive/negative valence split, relative to the median")
    ap.add_argument("--deadband", type=float, default=0.20,
                    help="radius from baseline inside which the state is neutral")
    args = ap.parse_args()

    d = load(args.csv)
    t, n = d["t"], len(d["t"])
    fix = ~np.isnan(d["pos"][:, 0])
    both = (d["det_l"] == 1) & (d["det_r"] == 1)
    one = ((d["det_l"] == 1) ^ (d["det_r"] == 1))
    rate = n / (t[-1] - t[0]) if n > 1 and t[-1] > t[0] else float("nan")

    nod, shake = detect_head_gestures(t, d["yaw"], d["pitch"],
                                      win=args.gesture_window,
                                      min_amp=args.gesture_amp)

    ipd = d["ipd"]
    ipd_ok = np.isfinite(ipd)
    ipd_med = float(np.nanmedian(ipd)) if ipd_ok.any() else float("nan")

    state, (a_base, v_base) = classify_affect(
        d["arousal"], d["valence"], a_thresh=args.a_thresh,
        v_thresh=args.v_thresh, deadband=args.deadband,
        have_face=np.isfinite(d["yaw"]))

    fig, ax = plt.subplots(4, 1, figsize=(15, 13), sharex=True)
    title = (f"{args.csv.name} — {n} rows, {rate:.1f} Hz, "
             f"3-D fix {100*fix.mean():.0f}%, one-eye-only {100*one.mean():.0f}%")
    if np.isfinite(ipd_med):
        title += f", median IPD {ipd_med*1000:.1f} mm"
    fig.suptitle(title, fontsize=12)

    # --- 1. detection + IPD ------------------------------------------------
    # IPD shares this panel because it is a statement about the same thing: not
    # whether a fix EXISTS but whether it is worth anything. Per-eye rows are
    # gone; "one eye only" already carries the part that changes a decision.
    a = ax[0]
    a.fill_between(t, 0.55, 1.0, where=fix, step="mid", color="tab:green",
                   alpha=.75, label="3-D fix")
    if one.any():
        a.fill_between(t, 0.0, 0.45, where=one, step="mid", color="tab:red",
                       alpha=.85, label="one eye only")
    a.set_yticks([0.225, 0.775])
    a.set_yticklabels(["1-eye", "3-D"])
    a.set_ylim(-0.1, 1.1)
    a.set_ylabel("detection")
    a.grid(alpha=.3, axis="x")
    a.legend(**LEGEND_TWIN)

    a2 = a.twinx()
    a2.axhspan(IPD_LO * 1000, IPD_HI * 1000, color="tab:green", alpha=.10)
    a2.plot(t, ipd * 1000, color="tab:red", lw=1.1, label="IPD")
    a2.set_ylabel("IPD (mm)", color="tab:red")
    a2.tick_params(axis="y", labelcolor="tab:red")
    if np.isfinite(ipd_med):
        a2.axhline(ipd_med * 1000, color="k", lw=.9, ls="--", alpha=.7)
        spread = float(np.nanpercentile(ipd, 95) - np.nanpercentile(ipd, 5)) * 1000
        a2.set_ylim(min(IPD_LO * 1000, np.nanmin(ipd) * 1000) - 3,
                    max(IPD_HI * 1000, np.nanmax(ipd) * 1000) + 3)
        # the 5-95 spread is the honest error bar on every position below
        a.text(0.005, 0.06, f"IPD median {ipd_med*1000:.1f} mm, 5-95 spread "
                            f"{spread:.1f} mm — the error bar on head position",
               transform=a.transAxes, fontsize=8.5, color="0.25")

    # --- 2. position -------------------------------------------------------
    a = ax[1]
    for i, (lab, colr) in enumerate([("X", "tab:blue"), ("Y", "tab:orange"),
                                     ("Z", "tab:green")]):
        a.plot(t, d["pos"][:, i], color=colr, lw=1.2, label=lab)
    a.plot(t, d["dist"], color="k", lw=1.6, label="distance")
    a.set_ylabel("head position (m)")
    a.grid(alpha=.3)
    a.legend(**LEGEND_TWIN)

    a2 = a.twinx()
    a2.plot(t, d["approach"], color="tab:purple", lw=1.0, alpha=.7,
            label="approach")
    a2.axhline(0, color="tab:purple", lw=.6, ls=":", alpha=.5)
    a2.set_ylabel("approach (m/s)", color="tab:purple")
    a2.tick_params(axis="y", labelcolor="tab:purple")

    # --- 3. head pose ------------------------------------------------------
    # Same two signals attention is derived from, shown raw. Two things to read
    # here: whether the attending ribbon tracks the cone (chatter at the
    # boundary means the threshold needs hysteresis, not that the person moved),
    # and OSCILLATION -- a nod is pitch swinging at 1-3 Hz, a shake is yaw.
    a = ax[2]
    for span in _spans(t, d["attending"] == 1):
        a.axvspan(*span, color="tab:green", alpha=.10)
    a.axhspan(-ATTEND_YAW_DEG, ATTEND_YAW_DEG, color="tab:blue", alpha=.06)

    # nod/shake bands, bounded in y so they do not bury the traces beneath them
    lo_b, hi_b = GESTURE_BAND
    for mask, lab in ((nod, "nod"), (shake, "shake")):
        if mask.any():
            a.fill_between(t, lo_b, hi_b, where=mask, step="mid",
                           color="tab:pink", alpha=.30, lw=0,
                           label=f"{lab} ({len(_spans(t, mask))})")
            for x0, x1 in _spans(t, mask):
                a.text((x0 + x1) / 2, hi_b, lab, ha="center", va="bottom",
                       fontsize=8, color="palevioletred")
    for lim, colr in ((ATTEND_YAW_DEG, "tab:blue"), (ATTEND_PITCH_DEG, "tab:orange")):
        for sign in (-1, 1):
            a.axhline(sign * lim, color=colr, lw=.8, ls=":", alpha=.7)
    a.plot(t, d["yaw"], color="tab:blue", lw=1.3, label=f"yaw (±{ATTEND_YAW_DEG:.0f}°)")
    a.plot(t, d["pitch"], color="tab:orange", lw=1.3,
           label=f"pitch (±{ATTEND_PITCH_DEG:.0f}°)")
    a.plot(t, d["roll"], color="tab:gray", lw=1.0, alpha=.7, label="roll")
    a.axhline(0, color="k", lw=.6, alpha=.4)
    a.set_ylabel("head pose (deg)")
    a.grid(alpha=.3)
    a.legend(**LEGEND_KW)

    # --- 4. affect + derived state ----------------------------------------
    # Shading is the 5-state classification, not attention -- attention already
    # has a ribbon on the pose panel above, and repeating it here would spend
    # the only channel this panel has left on information already shown.
    a = ax[3]
    seen = []
    for st in STATE_NAMES:
        mask = (state == st)
        if not mask.any():
            continue
        seen.append((st, 100.0 * mask.mean()))
        for x0, x1 in _spans(t, mask):
            a.axvspan(x0, x1, color=STATE_COLORS[st], alpha=.16, lw=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATE_COLORS[s], alpha=.45)
               for s, _ in seen]

    a.plot(t, d["arousal"], color="tab:blue", lw=1.3, label="arousal")
    a.plot(t, d["valence"], color="tab:green", lw=1.3, label="valence")
    a.plot(t, d["change"], color="tab:orange", lw=1.0, alpha=.8, label="change /s")
    # baselines the state test is measured FROM, not zero
    a.axhline(a_base, color="tab:blue", lw=.7, ls=":", alpha=.6)
    a.axhline(v_base, color="tab:green", lw=.7, ls=":", alpha=.6)
    a.axhline(0, color="k", lw=.6, alpha=.3)
    a.set_ylabel("affect  (shaded = state)")
    a.set_xlabel("time (s)")
    a.grid(alpha=.3)

    lines, labels = a.get_legend_handles_labels()
    a.legend(lines + handles,
             labels + [f"{s} {p:.0f}%" for s, p in seen], **LEGEND_TWIN)

    a2 = a.twinx()
    a2.plot(t, d["dwell"], color="tab:gray", lw=1.0, ls="--", label="dwell")
    a2.set_ylabel("dwell (s)", color="tab:gray")
    a2.tick_params(axis="y", labelcolor="tab:gray")

    fig.tight_layout(rect=[0, 0, 0.86, 0.97])

    # --- what the numbers say ---------------------------------------------
    print(f"  rows {n}   {rate:.1f} Hz   detect {np.nanmean(d['det_ms']):.1f} ms")
    print(f"  3-D fix {100*fix.mean():5.1f}%   both eyes {100*both.mean():5.1f}%"
          f"   one eye only {100*one.mean():5.1f}%")
    if np.isfinite(ipd_med):
        spread = float(np.nanpercentile(ipd, 95) - np.nanpercentile(ipd, 5))
        verdict = ("plausible" if IPD_LO <= ipd_med <= IPD_HI else
                   "IMPLAUSIBLE — stereo scale is wrong at face range")
        print(f"  IPD median {ipd_med*1000:5.1f} mm ({verdict}), "
              f"5-95 spread {spread*1000:.1f} mm")
    else:
        print("  IPD: never measured — no frame had both irises triangulated")
    if fix.any():
        print(f"  distance {np.nanmin(d['dist']):.2f}-{np.nanmax(d['dist']):.2f} m")
    for k in ("yaw", "pitch"):
        v = d[k][np.isfinite(d[k])]
        if v.size:
            print(f"  {k:8} range {v.min():+6.1f}..{v.max():+6.1f} deg  sd {v.std():.1f}")
    n_nod, n_shake = len(_spans(t, nod)), len(_spans(t, shake))
    print(f"  head gestures: {n_nod} nod, {n_shake} shake "
          f"(window {args.gesture_window}s, min swing {args.gesture_amp}deg)")
    att = np.nanmean(d["attending"]) if n else 0.0
    print(f"  attending {100*att:.0f}% of frames, "
          f"longest dwell {np.nanmax(d['dwell']):.1f} s")
    for k, lab in (("arousal", "arousal"), ("valence", "valence")):
        v = d[k]
        print(f"  {lab:8} range {np.nanmin(v):+.2f}..{np.nanmax(v):+.2f}  "
              f"sd {np.nanstd(v):.3f}")
    print(f"  affect baseline: arousal {a_base:+.3f}, valence {v_base:+.3f} "
          f"(take median)   deadband {args.deadband}")
    occ = "  ".join(f"{s} {100.0*(state == s).mean():.0f}%"
                    for s in STATE_NAMES if (state == s).any())
    print(f"  state occupancy: {occ}")
    flips = int(np.count_nonzero(state[1:] != state[:-1]))
    print(f"  state changes: {flips} in {t[-1]-t[0]:.0f}s "
          f"({flips/max(t[-1]-t[0], 1e-9):.2f}/s)")

    if args.save:
        fig.savefig(args.save, dpi=130)
        print(f"  -> {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
