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

from dataclasses import dataclass

from .config import DEFAULT, STATE_QUADRANT, AffectConfig


@dataclass(frozen=True)
class Latch:
    """What the classifier must remember between frames, and nothing else.

    Three bits: whether we were inside the deadband, and which side of each
    axis we were last on. The sign bits exist so a boundary crossing is STICKY
    rather than instantaneous -- without them a point sitting on an axis
    relabels itself on noise, forever.
    """
    in_neutral: bool = True
    hi_a: bool = False
    pos_v: bool = False


def quadrant(da: float, dv: float, latch: Latch = Latch(),
             cfg: AffectConfig = DEFAULT) -> tuple[str, Latch]:
    """Baseline-corrected (arousal, valence) -> (state, latch for next frame).

    Pure: no buffers, no timestamps, no arrays. Scalars in, answer out.

    A non-finite input yields 'unknown'. That guard lives HERE, in the rule,
    rather than in either driver, because it is a precondition of the rule --
    put it in the callers and each one has to remember it. One did not: the
    streaming path checked only whether a face was present, so a frame with a
    face but NaN blendshapes returned 'sad'. NaN fails every comparison, so it
    escaped the neutral radius test and fell into the (False, False) quadrant,
    and the robot would have mirrored sadness at someone whose expression it
    could not read.

    Two independent sign tests, NOT a weighted sum of the two axes. Any single
    weighted combination projects the plane onto a line, which collides the
    diagonally opposite quadrants: with equal weights, angry (+arousal,
    -valence) and content (-arousal, +valence) both score zero.

    THREE boundaries, all hysteretic. The radius: hold neutral until it clears
    `deadband`, then hold the quadrant until it falls back below
    `deadband * hyst`. Inverting those makes the deadband easier to escape than
    to re-enter, which produces MORE chatter (measured: 44 flips against 8).

    And each SIGN, which previously had none -- outside the circle the quadrant
    came from bare comparisons, so a point on an axis flipped on noise. To
    become high-arousal you must clear `a_thresh + margin`; to stop being it you
    must fall below `a_thresh - margin`. Coming out of neutral there is no prior
    quadrant worth being sticky about, so the first decision uses the bare
    thresholds.
    """
    if not (np.isfinite(da) and np.isfinite(dv)):
        return "unknown", Latch(in_neutral=True)
    bar = cfg.deadband if latch.in_neutral else cfg.deadband * cfg.deadband_hyst
    if float(np.hypot(da, dv)) < bar:
        return "neutral", Latch(in_neutral=True, hi_a=latch.hi_a,
                                pos_v=latch.pos_v)
    m = cfg.sign_margin
    if latch.in_neutral:
        hi_a, pos_v = da > cfg.a_thresh, dv > cfg.v_thresh
    else:
        hi_a = da > cfg.a_thresh + m if not latch.hi_a else da > cfg.a_thresh - m
        pos_v = dv > cfg.v_thresh + m if not latch.pos_v else dv > cfg.v_thresh - m
    return STATE_QUADRANT[(hi_a, pos_v)], Latch(False, hi_a, pos_v)


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
        self._latch = Latch()

    def update(self, t: float, arousal: float, valence: float,
               have_face: bool) -> str:
        if not have_face:
            self.state, self._latch = "unknown", Latch()
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
        self.state, self._latch = quadrant(
            arousal - a_base, valence - v_base, self._latch, self.cfg)
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
    latch = Latch()
    for i in range(len(arousal)):
        if not have_face[i] or not (np.isfinite(da[i]) and np.isfinite(dv[i])):
            out[i], latch = "unknown", Latch()
            continue
        out[i], latch = quadrant(da[i], dv[i], latch, cfg)
    return out, (a_base, v_base)
