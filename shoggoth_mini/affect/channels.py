"""Derived signals with TIME in them.

A single frame can say "a face is at 0.6 m looking slightly left". Only a
history can say "they have been watching for four seconds and are leaning in",
which is the kind of thing a robot should react to. Everything here is stateful
on purpose; the stateless per-frame quantities live in features.py.

TRAIL_MAX bounds the tip trail a viewer might draw; nothing here needs it.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .features import affect_from_blend, attending

TRAIL_MAX = 90


class Channels:
    """Turns per-frame observations into the signals an appraisal layer wants.

    Everything here is stateful on purpose. A single frame can say "a face is
    at 0.6 m looking slightly left"; only a history can say "they have been
    watching for four seconds and are leaning in", which is the kind of thing a
    robot should react to.
    """

    def __init__(self, smooth: float = 0.25):
        self.smooth = smooth
        self.pos: Optional[np.ndarray] = None
        self.arousal = self.valence = 0.0
        self.dwell = 0.0            # s of continuous attended presence
        self.absent = 0.0           # s since the face was last seen
        self.approach = 0.0         # m/s, + = coming closer
        self.change = 0.0           # rate of expression change
        self.trail: deque[np.ndarray] = deque(maxlen=TRAIL_MAX)
        self._prev_dist: Optional[float] = None
        self._prev_af: Optional[np.ndarray] = None
        self._t: Optional[float] = None

    @staticmethod
    def _ema(old, new, a):
        return new if old is None else (1 - a) * old + a * new

    def update(self, t: float, pos, obs) -> None:
        dt = 0.0 if self._t is None else max(t - self._t, 1e-6)
        self._t = t

        if obs is None:
            self.absent += dt
            if self.absent > 0.5:       # brief dropouts are detector noise,
                self.dwell = 0.0        # not the person leaving
            return
        self.absent = 0.0

        a, v = affect_from_blend(obs.blend)
        self.arousal = self._ema(self.arousal, a, self.smooth)
        self.valence = self._ema(self.valence, v, self.smooth)

        af = np.array([a, v])
        if self._prev_af is not None and dt > 0:
            self.change = self._ema(
                self.change, float(np.linalg.norm(af - self._prev_af) / dt), 0.2)
        self._prev_af = af

        if pos is not None:
            self.pos = pos if self.pos is None else self._ema(self.pos, pos, self.smooth)
            self.trail.append(np.asarray(self.pos, float))
            d = float(np.linalg.norm(self.pos))
            if self._prev_dist is not None and dt > 0:
                self.approach = self._ema(
                    self.approach, (self._prev_dist - d) / dt, 0.2)
            self._prev_dist = d

        self.dwell = self.dwell + dt if self.attending(obs) else 0.0

    @staticmethod
    def attending(obs) -> bool:
        return attending(obs.yaw, obs.pitch)

    @property
    def distance(self) -> Optional[float]:
        return None if self.pos is None else float(np.linalg.norm(self.pos))
