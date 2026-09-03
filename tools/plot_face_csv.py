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
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shoggoth_mini.affect import (ATTEND_PITCH_DEG, ATTEND_YAW_DEG,  # noqa: E402
                                  STATE_COLORS, STATE_NAMES, classify_affect,
                                  detect_head_gestures)
from shoggoth_mini.affect.gestures import _spans  # noqa: E402

LEGEND_KW = dict(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1,
                 borderaxespad=0.0, frameon=False)
# Panels carrying a twinx need the legend pushed clear of the right-hand tick
# labels, which occupy the strip the default anchor sits in.
LEGEND_TWIN = dict(LEGEND_KW, bbox_to_anchor=(1.08, 1.0))

IPD_LO, IPD_HI = 0.055, 0.072      # plausible adult interpupillary distance (m)



def load_live_states(path):
    """The `state` column, if the take carries one. None for older recordings."""
    rows = list(csv.DictReader(open(path)))
    if not rows or "state" not in rows[0]:
        return None
    return np.array([(r["state"] or "").strip() for r in rows], dtype=object)


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
    ap.add_argument("--recompute", action="store_true",
                    help="ignore the take's own state column and reclassify from "
                         "arousal/valence using the whole take's median")
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

    # The LIVE labels are authoritative when the take has them: they are what
    # the robot actually acted on, computed from a running median that could
    # only see the past. Recomputing here uses the whole take's median, which is
    # a different (and unavailable-at-runtime) estimate -- on one real recording
    # the two agree on only 75% of frames. Showing the recomputed labels while
    # the state machine consumed the live ones makes transitions look
    # unmotivated for reasons that are entirely an artefact of this plot.
    live = None if args.recompute else load_live_states(args.csv)
    recomputed, (a_base, v_base) = classify_affect(
        d["arousal"], d["valence"], a_thresh=args.a_thresh,
        v_thresh=args.v_thresh, deadband=args.deadband,
        have_face=np.isfinite(d["yaw"]))
    if live is not None:
        state, shadow, shown_src = live, recomputed, "live (state column)"
    else:
        state, shadow, shown_src = recomputed, None, "recomputed here"

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
    for st in sorted(set(state) | set(STATE_NAMES), key=lambda x: (
            STATE_NAMES.index(x) if x in STATE_NAMES else 99)):
        mask = (state == st)
        if not mask.any():
            continue
        seen.append((st, 100.0 * mask.mean()))
        for x0, x1 in _spans(t, mask):
            a.axvspan(x0, x1, color=STATE_COLORS.get(st, "0.5"), alpha=.16, lw=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATE_COLORS.get(s, "0.5"),
                             alpha=.45) for s, _ in seen]

    # Shadow ribbon: the OTHER classification, drawn as a thin strip along the
    # bottom. Where it disagrees with the shading above, the running baseline
    # and the whole-take baseline reached different conclusions -- which is a
    # diagnostic for BASELINE_WINDOW_S, not a bug in either.
    if shadow is not None:
        tr = mtransforms.blended_transform_factory(a.transData, a.transAxes)
        for st in set(shadow):
            for x0, x1 in _spans(t, shadow == st):
                a.fill_between([x0, x1], 0.0, 0.045,
                               color=STATE_COLORS.get(st, "0.5"), alpha=.85,
                               lw=0, transform=tr)
        agree = 100.0 * float((state == shadow).mean())
        a.text(0.004, 0.955, f"lower strip: recomputed here — agrees with live "
                             f"on {agree:.0f}% of frames",
               transform=a.transAxes, fontsize=8, color="0.3")

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
    print(f"  affect labels: {shown_src}")
    if shadow is not None:
        print(f"    agreement with the recomputed labels: "
              f"{100*float((state == shadow).mean()):.1f}%")
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
