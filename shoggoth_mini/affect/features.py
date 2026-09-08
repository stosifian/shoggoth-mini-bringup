"""Per-frame quantities derived from one face observation.

Stateless by construction: everything here answers a question about a single
frame. Anything needing history lives in state.py, and the split is deliberate
-- it is what lets the same functions serve the live loop, the offline plots and
the state-machine checker without any of them disagreeing.
"""

from __future__ import annotations

import numpy as np

from .config import DEFAULT, AffectConfig


def head_angles(matrix) -> tuple[float, float, float, np.ndarray]:
    """Yaw/pitch/roll (deg) + forward axis from a 4x4 facial transform.

    Derived from the rotation's third column rather than a full Euler
    decomposition: we only need a pointing direction, and a forward vector
    degrades gracefully near gimbal configurations where Euler angles do not.

    SIGNS ARE UNVERIFIED against a real rig -- they depend on MediaPipe's
    convention and on how the cameras are mounted. Turn right and check yaw goes
    positive; nod down and check pitch goes positive.
    """
    R = np.asarray(matrix, float)[:3, :3]
    fwd = R[:, 2]
    yaw = float(np.degrees(np.arctan2(fwd[0], abs(fwd[2]) + 1e-9)))
    pitch = float(np.degrees(np.arcsin(np.clip(-fwd[1], -1.0, 1.0))))
    roll = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return yaw, pitch, roll, fwd


def affect_from_blend(blend: dict, cfg: AffectConfig = DEFAULT) -> tuple[float, float]:
    """Weighted blendshape sums -> (arousal, valence), each roughly in [-1, 1].

    Normalised by the total weight mass so the scale does not shift every time a
    term is added to the table, then scaled by a gain that is frankly arbitrary
    and worth tuning against a real face -- a full smile currently reads ~+0.97.
    """
    out = []
    for axis in ("arousal", "valence"):
        w = cfg.blendshape_weights[axis]
        norm = sum(abs(v) for v in w.values()) or 1.0
        out.append(float(np.clip(
            sum(blend.get(k, 0.0) * v for k, v in w.items()) / norm
            * cfg.blendshape_gain, -1.0, 1.0)))
    return out[0], out[1]


def attending(yaw: float, pitch: float, latched: bool = False,
              cfg: AffectConfig = DEFAULT) -> bool:
    """Is the head pointed close enough to head-on to count as looking at us?

    Head pose, NOT gaze: someone can look at the robot out of the corner of
    their eye and register as not attending. The iris landmarks exist to fix
    that later; nothing uses them yet.

    A SCHMITT TRIGGER, not a bare comparison. `latched` is the previous answer:
    entering costs margin, leaving refunds it, so the thresholds are 23/18 deg
    on the way in and 27/22 on the way out.

    The bare comparison this replaces chattered, and the cost was measured
    rather than assumed. In a 155 s session the attention bit flipped 64 times,
    and 57 of 61 flips happened within ONE degree of a threshold (median 0.28
    deg) -- a head parked near the boundary being toggled by 0.4-0.9 deg of
    landmark noise. Every flip resets a dwell timer that may have been building
    for ten seconds.

    Note that no amount of input smoothing fixes this. Filtering makes a head
    parked exactly on the threshold chatter more slowly, not less; only a gap
    between the entry and exit tests removes it. That is why this is here and
    not in Channels' EMA.
    """
    m = cfg.attend_margin_deg if latched else -cfg.attend_margin_deg
    return (abs(yaw) < cfg.attend_yaw_deg + m
            and abs(pitch) < cfg.attend_pitch_deg + m)
