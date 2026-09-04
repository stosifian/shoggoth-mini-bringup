"""Every threshold, weight and palette the affect stack uses, in one place.

These were previously spread across face_probe.py and plot_face_csv.py, some as
module constants and some as function defaults, with a comment in one file
asking the reader to keep it in step with the other. That is a convention, not
a guarantee, and it had already drifted once.

Nothing here is measured or derived -- these are choices. Where a value came
from an experiment the comment says so; where it is a guess, it says that too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- attention ---------------------------------------------------------------
# Head-pose cone counted as "looking at the robot". Deliberately asymmetric:
# people scan horizontally more than vertically. Neither number has been checked
# against anyone actually looking at the robot.
ATTEND_YAW_DEG = 25.0
ATTEND_PITCH_DEG = 20.0

# --- affect classification ---------------------------------------------------
# Splits are RELATIVE to a baseline, not absolute. A resting face is not (0, 0):
# arousal here is built from brow raise, eye widening and jaw open, all of which
# are additions to a neutral face, so its resting value sits near the bottom of
# its range rather than in the middle.
A_THRESH = 0.0
V_THRESH = 0.0
DEADBAND = 0.20          # radius from baseline inside which the state is neutral
DEADBAND_HYST = 0.65     # leave neutral at DEADBAND, return at 0.65 x that
# Hysteresis on the two SIGN tests, which previously had none: outside the
# deadband the quadrant was decided by bare comparisons, so a point parked on an
# axis flipped between neighbouring quadrants on noise. Measured over a 96 s
# session, 17 of 28 label changes crossed a quadrant boundary rather than the
# deadband, twelve of them sad<->angry, with |d_arousal| under 0.02 on 28% of
# frames outside the circle. Arousal is the axis this bites: it is built from
# additions to a neutral face, so it rarely goes negative and hovers near zero.
SIGN_MARGIN = 0.04
BASELINE_WINDOW_S = 20.0  # running-median window, live path only
BASELINE_MIN_S = 2.0     # below this a median is noise, so report 'calibrating'

STATE_QUADRANT = {(False, True): "content", (True, True): "excited",
                  (False, False): "sad", (True, False): "angry"}

# 'unknown' is not a sixth quadrant, it is the absence of an observation: no
# face, so nothing to classify. Kept distinct from neutral because a state
# machine rule reading "Neutral == 1" must not fire when someone left the room.
STATE_NAMES = ["unknown", "neutral", "content", "excited", "sad", "angry"]

# Two renderers, one meaning. Warm = high arousal, cool = low, grey = no claim.
STATE_COLORS = {"unknown": "0.85", "neutral": "0.62", "content": "tab:green",
                "excited": "gold", "sad": "tab:blue", "angry": "tab:red"}
STATE_BGR = {"neutral": (158, 158, 158), "content": (120, 180, 100),
             "excited": (0, 215, 255), "sad": (180, 119, 31),
             "angry": (40, 39, 214), "calibrating": (110, 110, 110),
             # dimmer than neutral's grey on purpose: "nobody there" and "calm
             # person" must not look the same at a glance
             "unknown": (78, 78, 78)}

# --- head gestures -----------------------------------------------------------
GESTURE_HZ = (0.8, 4.0)     # a nod or shake, in cycles per second
MIN_EVENT_S = 0.30          # shorter than this is not a gesture
MERGE_GAP_S = 0.30          # detections closer than this are one event
GESTURE_MIN_AMP = 8.0       # floor on peak-to-peak swing, degrees
GESTURE_WINDOW_S = 1.0      # sliding window a gesture is measured in
GESTURE_MAX_YAW = 30.0      # a gesture is aimed AT the robot; beyond this it is not
NOISE_WINDOW_S = 5.0        # neighbourhood the local noise floor is measured over

# --- blendshapes -> affect ----------------------------------------------------
# A STARTING GUESS, deliberately legible and tunable rather than learned: these
# are facial GEOMETRY activations, and the step from geometry to emotion is an
# interpretation that belongs where it can be tuned by hand. Single-frame
# categorical emotion recognition is unreliable in the wild anyway; what earns
# its keep is CHANGE over time.
BLENDSHAPE_WEIGHTS = {
    "valence": {"mouthSmileLeft": +1.0, "mouthSmileRight": +1.0,
                "cheekSquintLeft": +0.5, "cheekSquintRight": +0.5,
                "mouthFrownLeft": -1.0, "mouthFrownRight": -1.0,
                "browDownLeft": -0.6, "browDownRight": -0.6},
    "arousal": {"eyeWideLeft": +0.8, "eyeWideRight": +0.8,
                "browInnerUp": +0.7,
                "browOuterUpLeft": +0.5, "browOuterUpRight": +0.5,
                "jawOpen": +0.6,
                "mouthStretchLeft": +0.3, "mouthStretchRight": +0.3},
}
BLENDSHAPE_GAIN = 3.0       # scales the normalised weighted sum into [-1, 1]


@dataclass(frozen=True)
class AffectConfig:
    """Bundle passed to the classifier so a caller can vary it without globals."""
    a_thresh: float = A_THRESH
    v_thresh: float = V_THRESH
    deadband: float = DEADBAND
    deadband_hyst: float = DEADBAND_HYST
    sign_margin: float = SIGN_MARGIN
    baseline_window_s: float = BASELINE_WINDOW_S
    baseline_min_s: float = BASELINE_MIN_S
    attend_yaw_deg: float = ATTEND_YAW_DEG
    attend_pitch_deg: float = ATTEND_PITCH_DEG
    blendshape_weights: dict = field(default_factory=lambda: BLENDSHAPE_WEIGHTS)
    blendshape_gain: float = BLENDSHAPE_GAIN


DEFAULT = AffectConfig()
