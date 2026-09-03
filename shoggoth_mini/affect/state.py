"""Affect classification: one rule, two drivers.

The rule used to exist twice -- once inside a stateful streaming class and once
inside a batch loop -- and the two were verified to agree only by test. Here the
decision is a single pure function that both call, so agreement is structural
rather than something to re-check.

The ONLY difference between live and offline is the BASELINE, and that
difference is unavoidable rather than accidental: a running median can see only
the past, a whole-take median can see everything. On one real recording the two
label 75% of frames identically. Which is right depends on the question. If you
are asking what the robot did, the running baseline is the only honest answer,
because it is what the robot had.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .config import DEFAULT, STATE_QUADRANT, AffectConfig


def quadrant(da: float, dv: float, in_neutral: bool,
             cfg: AffectConfig = DEFAULT) -> tuple[str, bool]:
    """Baseline-corrected (arousal, valence) -> (state, still_in_neutral).

    Pure: no buffers, no timestamps, no arrays. Scalars in, answer out.

    Two independent sign tests, NOT a weighted sum of the two axes. Any single
    weighted combination projects the plane onto a line, which collides the
    diagonally opposite quadrants: with equal weights, angry (+arousal,
    -valence) and content (-arousal, +valence) both score zero.

    `neutral` is a radius test rather than a fifth quadrant -- close enough to
    baseline that the quadrant is noise. It carries hysteresis, and the
    direction matters: hold neutral until the radius clears `deadband`, then
    hold the quadrant until it falls back below `deadband * hyst`. Inverting
    those two makes the deadband easier to escape than to re-enter, which
    produces MORE chatter, not less (measured: 44 flips against 8).
    """
    bar = cfg.deadband if in_neutral else cfg.deadband * cfg.deadband_hyst
    if float(np.hypot(da, dv)) < bar:
        return "neutral", True
    return STATE_QUADRANT[(da > cfg.a_thresh, dv > cfg.v_thresh)], False


# =============================================================================
# streaming: what the robot has
# =============================================================================
class AffectState:
    """Live affect label from a running baseline.

    Reports 'unknown' with no face -- neutral is a claim about a face that was
    looked at, and conflating the two makes an absence indistinguishable from a
    calm person. In the state machine's terms 'unknown' is all five emotion
    inputs at zero, not a sixth value.
    """

    def __init__(self, window_s: float | None = None,
                 cfg: AffectConfig = DEFAULT):
        self.cfg = cfg
        self.window_s = cfg.baseline_window_s if window_s is None else window_s
        self.buf: deque[tuple[float, float, float]] = deque()
        self.state = "calibrating"
        self.baseline = (0.0, 0.0)
        self._in_neutral = True

    def update(self, t: float, arousal: float, valence: float,
               have_face: bool) -> str:
        if not have_face:
            self.state, self._in_neutral = "unknown", True
            return self.state

        self.buf.append((t, arousal, valence))
        while self.buf and t - self.buf[0][0] > self.window_s:
            self.buf.popleft()

        span = self.buf[-1][0] - self.buf[0][0] if len(self.buf) > 1 else 0.0
        if span < self.cfg.baseline_min_s:
            self.state = "calibrating"      # a median over half a second is noise
            return self.state

        arr = np.asarray(self.buf, float)
        a_base, v_base = float(np.median(arr[:, 1])), float(np.median(arr[:, 2]))
        self.baseline = (a_base, v_base)
        self.state, self._in_neutral = quadrant(
            arousal - a_base, valence - v_base, self._in_neutral, self.cfg)
        return self.state


# =============================================================================
# batch: what a whole recording looks like in hindsight
# =============================================================================
def classify_affect(arousal, valence, a_thresh=None, v_thresh=None,
                    deadband=None, hyst=None, have_face=None,
                    cfg: AffectConfig = DEFAULT):
    """Label a whole take at once, using its own median as the baseline.

    `have_face` marks frames with no detection; those are 'unknown'. Baselines
    are computed only over frames that HAVE a face, so a take that is mostly
    empty room does not drag the median toward whatever the affect signals read
    as when there is nothing to read.
    """
    from dataclasses import replace
    over = {k: v for k, v in (("a_thresh", a_thresh), ("v_thresh", v_thresh),
                              ("deadband", deadband), ("deadband_hyst", hyst))
            if v is not None}
    if over:
        cfg = replace(cfg, **over)

    arousal = np.asarray(arousal, float)
    valence = np.asarray(valence, float)
    if have_face is None:
        have_face = np.ones(len(arousal), bool)
    have_face = np.asarray(have_face, bool)
    seen = have_face & np.isfinite(arousal) & np.isfinite(valence)

    a_base = float(np.median(arousal[seen])) if seen.any() else 0.0
    v_base = float(np.median(valence[seen])) if seen.any() else 0.0
    da, dv = arousal - a_base, valence - v_base

    out = np.full(len(arousal), "neutral", dtype=object)
    in_neutral = True
    for i in range(len(arousal)):
        if not have_face[i] or not (np.isfinite(da[i]) and np.isfinite(dv[i])):
            out[i], in_neutral = "unknown", True
            continue
        out[i], in_neutral = quadrant(da[i], dv[i], in_neutral, cfg)
    return out, (a_base, v_base)
