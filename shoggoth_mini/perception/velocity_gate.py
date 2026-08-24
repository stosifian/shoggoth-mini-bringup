"""Reject physically impossible jumps in a tracked 3D point.

Detections occasionally land somewhere the object cannot have travelled to in one
frame — a mismatched stereo correspondence, or the detector latching onto the wrong
thing. Measured on this build: ~20% of tip estimates moved more than 50 mm between
consecutive frames at ~16 Hz, which the tentacle cannot physically do.

These are worse than a missing detection. A missing point makes the control loop
skip the iteration; a wrong point steers the robot. And because the observation
stacks 4 frames, one bad estimate contaminates ~250 ms of policy input.

Design decisions worth knowing:

* A rejection returns None, NOT the last good position. The control loop already
  handles a missing point; feeding it a stale one disguised as fresh is worse,
  because nothing downstream can tell the difference.
* Two escape hatches are mandatory, not optional. Without them a single bad
  reference position would lock out every subsequent detection forever:
    - after `max_rejects` consecutive rejections, accept anyway (the reference is
      probably the thing that was wrong)
    - if the reference is older than `stale_after_s`, accept unconditionally (dt
      is too long for a speed comparison to mean anything)
"""

import time
from typing import Optional

import numpy as np


class VelocityGate:
    """Speed-limit filter for one tracked point."""

    def __init__(
        self,
        max_speed_m_s: float,
        max_rejects: int = 3,
        stale_after_s: float = 0.5,
        name: str = "point",
    ):
        self.max_speed_m_s = float(max_speed_m_s)
        self.max_rejects = int(max_rejects)
        self.stale_after_s = float(stale_after_s)
        self.name = name

        self._last: Optional[np.ndarray] = None
        self._last_t: Optional[float] = None
        self._consecutive_rejects = 0

        self.n_accepted = 0
        self.n_rejected = 0
        self.n_forced = 0
        self.last_was_rejected = False

    def reset(self) -> None:
        self._last = None
        self._last_t = None
        self._consecutive_rejects = 0

    def update(
        self, point: Optional[np.ndarray], now: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """Return the point if plausible, else None.

        Args:
            point: Newly triangulated position, or None if not detected.
            now: Timestamp override (used by offline replay).
        """
        self.last_was_rejected = False

        if point is None:
            # Nothing to judge. Leave the reference alone so a brief dropout does
            # not discard our notion of where the object was.
            return None

        point = np.asarray(point, dtype=float)
        now = time.time() if now is None else float(now)

        if self._last is None or self._last_t is None:
            self._accept(point, now)
            return point

        dt = now - self._last_t
        if dt <= 0 or dt > self.stale_after_s:
            # Reference too old (or clock went backwards) for a speed test.
            self._accept(point, now)
            return point

        speed = float(np.linalg.norm(point - self._last) / dt)
        if speed <= self.max_speed_m_s:
            self._accept(point, now)
            return point

        self._consecutive_rejects += 1
        if self._consecutive_rejects >= self.max_rejects:
            # Our reference is the more likely suspect now — adopt the new point.
            self._accept(point, now)
            self.n_forced += 1
            return point

        self.n_rejected += 1
        self.last_was_rejected = True
        return None

    def _accept(self, point: np.ndarray, now: float) -> None:
        self._last = point
        self._last_t = now
        self._consecutive_rejects = 0
        self.n_accepted += 1

    @property
    def stats(self) -> dict:
        total = self.n_accepted + self.n_rejected
        return {
            "name": self.name,
            "accepted": self.n_accepted,
            "rejected": self.n_rejected,
            "forced": self.n_forced,
            "reject_rate": (self.n_rejected / total) if total else 0.0,
        }


__all__ = ["VelocityGate"]
