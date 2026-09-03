"""Affect estimation: turning face geometry into something a robot can act on.

Interpretation, not sensing -- perception/ finds a face, this decides what to
make of it. The split matters because the orchestrator imports from here, and
so do the offline plots and the state-machine checker: one implementation, so
the robot and the tools that explain it cannot disagree.
"""

from .config import (ATTEND_PITCH_DEG, ATTEND_YAW_DEG, BLENDSHAPE_WEIGHTS,
                     DEFAULT, STATE_BGR, STATE_COLORS, STATE_NAMES,
                     STATE_QUADRANT, AffectConfig)
from .channels import Channels
from .features import affect_from_blend, attending, head_angles
from .gestures import _rolling_sigma, _spans, detect_head_gestures
from .fsm import Machine, load_tables, run_vectors
from .state import AffectState, classify_affect, quadrant

__all__ = ["AffectConfig", "BLENDSHAPE_WEIGHTS", "Channels", "DEFAULT", "STATE_NAMES", "STATE_COLORS", "STATE_BGR",
           "STATE_QUADRANT", "ATTEND_YAW_DEG", "ATTEND_PITCH_DEG",
           "head_angles", "affect_from_blend", "attending",
           "detect_head_gestures", "quadrant", "AffectState", "classify_affect",
           "_rolling_sigma", "_spans",
           "Machine", "load_tables", "run_vectors"]
