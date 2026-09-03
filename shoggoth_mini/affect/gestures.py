"""Nod and shake detection from the head-pose signals.

Lives here rather than in a plotting tool because three callers need identical
answers: the live probe, the offline plot, and the state-machine checker. It
used to live in plot_face_csv, which meant fsm_check imported it tool-from-tool.

No new inference -- this runs on yaw and pitch that the face landmarker already
produced.
"""

from __future__ import annotations

import numpy as np

from .config import (GESTURE_HZ, GESTURE_MAX_YAW, GESTURE_MIN_AMP,
                     GESTURE_WINDOW_S, MERGE_GAP_S, MIN_EVENT_S, NOISE_WINDOW_S)


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

def _rolling_sigma(t, sig, win_s=NOISE_WINDOW_S):
    """Per-frame noise floor, measured over a local neighbourhood.

    A single figure for a whole take is wrong when conditions change within it.
    Measured on one recording: landmark noise is 0.64 deg with the face head-on
    and 2.96 deg at 57 deg of yaw, because MediaPipe's fit degrades badly on a
    profile. A global estimate is dominated by the quiet stretches, so the bar
    ends up far too low exactly where the signal is worst -- which is how a
    profile-view wobble of 8.3 deg was reported as a head shake.

    The window is deliberately much longer than a gesture. Measure noise over a
    window the size of the gesture and a real shake inflates its own noise
    estimate until it fails its own test; at 5 s a 0.6 s gesture is ~12% of the
    samples, which the median absorbs.
    """
    t = np.asarray(t, float)
    d = np.abs(np.diff(np.asarray(sig, float)))
    out = np.zeros(len(t))
    for i in range(len(t)):
        lo = np.searchsorted(t, t[i] - win_s / 2.0, "left")
        hi = np.searchsorted(t, t[i] + win_s / 2.0, "right")
        seg = d[lo:max(hi - 1, lo + 1)]
        seg = seg[np.isfinite(seg)]
        out[i] = float(np.median(seg) / 1.35) if seg.size else 0.0
    return out

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

def detect_head_gestures(t, yaw, pitch, win=GESTURE_WINDOW_S,
                         min_amp=GESTURE_MIN_AMP, max_yaw=GESTURE_MAX_YAW):
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

    Two guards against the same false positive, both needed. A nod or shake is
    a communicative act AIMED at the robot, so a window whose mean yaw is
    outside `max_yaw` is not a gesture however much it oscillates -- gate on the
    window mean rather than per frame, since a real shake swings yaw and
    per-frame clipping would cut its own extremes off. And the amplitude bar
    comes from a LOCAL noise estimate, because the same wobble means different
    things head-on and in profile.
    """
    n = len(t)
    amp = {"nod": np.zeros(n), "shake": np.zeros(n)}
    # Both bars are set off the signal's own noise floor. A fixed threshold is
    # either deaf on a clean rig or, as measured here, permanently triggered on
    # a noisy one -- head-pose noise on a marginal detection is several degrees,
    # which is a good fraction of a real gesture.
    sigma = {"nod": _rolling_sigma(t, pitch), "shake": _rolling_sigma(t, yaw)}

    for i in range(n):
        lo = np.searchsorted(t, t[i] - win / 2.0, "left")
        hi = np.searchsorted(t, t[i] + win / 2.0, "right")
        if hi - lo < 5:
            continue
        # facing away is not addressing the robot, whatever the head is doing
        facing = yaw[lo:hi]
        facing = facing[np.isfinite(facing)]
        if facing.size and abs(float(facing.mean())) > max_yaw:
            continue
        span = float(t[hi - 1] - t[lo]) or win
        for key, sig in (("nod", pitch), ("shake", yaw)):
            w = sig[lo:hi]
            if not np.all(np.isfinite(w)):      # a dropout breaks the window
                continue
            w = w - w.mean()
            sg = float(sigma[key][i])
            ptp = float(np.ptp(w))
            if ptp < max(min_amp, 6.0 * sg):
                continue
            halves = _half_cycles(w, max(0.30 * ptp, 3.0 * sg, 2.0))
            if halves < 3:                      # under 1.5 swings is not a gesture
                continue
            if GESTURE_HZ[0] <= (halves / 2.0) / span <= GESTURE_HZ[1]:
                amp[key][i] = ptp

    nod = _clean(t, (amp["nod"] > 0) & (amp["nod"] >= amp["shake"]))
    shake = _clean(t, (amp["shake"] > 0) & (amp["shake"] > amp["nod"]))
    return nod, shake


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
