"""Stereo face perception bench -- landmarks, 3-D head position, expression channels.

DESIGN/RESEARCH (see NEXT_PHASE_EXPLORATION.md). Read-only w.r.t. the POC: this
imports the perception stack but touches no motor and writes nothing the robot
reads. Nothing here can move the tentacle.

What it is for: the ELEGNT framework (arXiv 2501.12493) splits expressive motion
into ATTENTION (where the robot looks), EMOTION (how it moves), INTENTION and
ATTITUDE. The first two need perception this project does not have yet. This
answers whether that perception is good enough to drive anything, BEFORE any of
it is allowed near a servo.

    python exploration/face_probe.py                    # live, both eyes + panel
    python exploration/face_probe.py --headless --csv out/face.csv
    python exploration/face_probe.py --source clip.mp4  # no camera contention
    python exploration/face_probe.py --flip-yaw         # see 'head pose' below

MediaPipe is one library call. The part being prototyped is everything after it:

  * TRIANGULATION. Landmarks are normalised per-eye and scale-free. Running the
    two views through the project's own stereo calibration is what turns a face
    into a referent in METRES -- which is what "someone leaned in" needs to be a
    measurable event rather than a guess.
  * DERIVED CHANNELS. arousal/valence are weighted blendshape sums. The weights
    below are a STARTING GUESS, not a validated affect model -- see the note on
    BLENDSHAPE_WEIGHTS.
  * TIME. Single-frame values are close to useless for the appraisal layer.
    Dwell, approach speed and expression CHANGE are the signals with content.

Sanity check worth watching: the panel reports measured interpupillary distance.
A human adult IPD is ~63 mm (range roughly 55-72). If the number on screen is
far outside that, the stereo scale is wrong AT FACE RANGE and every 3-D value
here is wrong with it -- the calibration was fitted around the tentacle, much
closer to the cameras than a person's head sits.

HEAD POSE IS UNVERIFIED. The yaw/pitch signs depend on the sign convention of
MediaPipe's transformation matrix and on how the rig is mounted, and could not
be checked without a face in front of the cameras. Turn your head right and
confirm yaw goes positive; nod down and confirm pitch goes positive. If either
is backwards, pass --flip-yaw / --flip-pitch and record it in the config.
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

from shoggoth_mini.perception.camera import read_oriented          # noqa: E402
from shoggoth_mini.perception.stereo import (                      # noqa: E402
    load_stereo_calibration,
    split_stereo_frame,
    triangulate_points,
)
from shoggoth_mini.configs import get_perception_config            # noqa: E402

MODEL_PATH = PKG / "assets" / "models" / "vision" / "face_landmarker.task"

# The project YAML must be named EXPLICITLY. get_perception_config() with no
# argument returns pydantic field defaults and silently ignores it -- the trap
# perception/camera.py documents. Getting this wrong here cost a session: the
# default units_to_meters is 0.05 against the YAML's 1.0, so every triangulated
# distance came out 20x small and a 63 mm IPD read as 3.1 mm.
DEFAULT_PERCEPTION_YAML = PKG / "shoggoth_mini" / "configs" / "default_perception.yaml"

# MediaPipe's face detector runs on a ~192 px square. A full 1920x760 eye is
# therefore scaled by 0.1, so a 150 mm face at 0.5 m (186 px at fx=620) reaches
# the detector as ~19 px -- right at BlazeFace's floor, which is why detection
# dies just past half a metre. Cropping to a square recovers the pixels: 760 px
# gives 0.25 scale and ~47 px on the same face.
SEARCH_CROP = 760
ROI_MIN = 160
ROI_MARGIN = 2.2            # crop side as a multiple of the face's larger extent

# 478-landmark layout: 0..467 mesh, 468..472 left iris, 473..477 right iris.
# "Left" is MediaPipe's naming (the subject's left), not the image side.
NOSE_TIP = 1
IRIS_L, IRIS_R = 468, 473

# Face-range triangulation box. triangulate_points CLIPS to coordinate_limits
# and indexes it unconditionally, so None is not an option -- and the project's
# configured limits are the tentacle's workspace, a box a face never enters.
# Clipping to it would silently pin every head position to a corner.
FACE_LIMITS = {"X": {"clip_min": -3.0, "clip_max": 3.0},
               "Y": {"clip_min": -3.0, "clip_max": 3.0},
               "Z": {"clip_min": -3.0, "clip_max": 3.0}}

# Blendshape -> affect. A STARTING GUESS, deliberately legible and tunable
# rather than learned: these are facial GEOMETRY activations, and the step from
# geometry to emotion is an interpretation that belongs in the appraisal layer
# where it can be tuned by hand. Single-frame categorical emotion recognition is
# unreliable in the wild anyway; what earns its keep here is CHANGE over time.
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

ATTEND_YAW_DEG = 25.0       # within this of head-on counts as "looking at it"
ATTEND_PITCH_DEG = 20.0

# Affect state. Must match classify_affect() in tools/plot_face_csv.py, so a
# take labelled live and the same take replotted afterwards agree.
A_THRESH = 0.0              # high/low arousal split, relative to baseline
V_THRESH = 0.0              # positive/negative valence split, same
DEADBAND = 0.20             # radius from baseline inside which state is neutral
DEADBAND_HYST = 0.65        # leave neutral at DEADBAND, return at 0.65 x that
BASELINE_WINDOW_S = 20.0    # running median window
BASELINE_MIN_S = 2.0        # below this the baseline is not worth trusting

STATE_QUADRANT = {(False, True): "content", (True, True): "excited",
                  (False, False): "sad", (True, False): "angry"}
# BGR, to match the plotter's palette: warm = high arousal, cool = low
STATE_BGR = {"neutral": (158, 158, 158), "content": (120, 180, 100),
             "excited": (0, 215, 255), "sad": (180, 119, 31),
             "angry": (40, 39, 214), "calibrating": (110, 110, 110),
             # deliberately dimmer than neutral's grey: "nobody there" and
             # "calm person" must not look the same at a glance
             "unknown": (78, 78, 78)}
IPD_NOMINAL_M = 0.063
TRAIL_MAX = 90

_thread_local = threading.local()


# =============================================================================
# detection
# =============================================================================
def _detector():
    """Thread-local FaceLandmarker.

    Same pattern as perception/hand_tracking.py: the detector is not thread
    safe, and the two eyes are processed concurrently.
    """
    d = getattr(_thread_local, "face", None)
    if d is not None:
        return d
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker, FaceLandmarkerOptions, RunningMode)

    if not MODEL_PATH.exists():
        sys.exit(
            f"face_landmarker.task not found at {MODEL_PATH}\n"
            "  curl -o assets/models/vision/face_landmarker.task \\\n"
            "    https://storage.googleapis.com/mediapipe-models/face_landmarker"
            "/face_landmarker/float16/1/face_landmarker.task")

    d = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True))
    _thread_local.face = d
    return d


@dataclass
class FaceObs:
    """One eye's view of one face."""
    px: np.ndarray                  # (478, 2) landmark pixels
    blend: dict                     # blendshape name -> score
    yaw: float = 0.0                # degrees, + = turned one way (see docstring)
    pitch: float = 0.0
    roll: float = 0.0
    fwd: np.ndarray = field(default_factory=lambda: np.zeros(3))


def head_angles(matrix: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Yaw/pitch/roll (deg) + forward axis from the 4x4 facial transform.

    Derived from the rotation's third column rather than a full Euler
    decomposition: we only need a pointing direction, and a forward vector
    degrades gracefully near gimbal configurations where Euler angles do not.
    """
    R = np.asarray(matrix, float)[:3, :3]
    fwd = R[:, 2]
    yaw = float(np.degrees(np.arctan2(fwd[0], abs(fwd[2]) + 1e-9)))
    pitch = float(np.degrees(np.arcsin(np.clip(-fwd[1], -1.0, 1.0))))
    roll = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return yaw, pitch, roll, fwd


class Roi:
    """Per-eye square crop that follows the face.

    Two jobs. It keeps enough pixels ON the face for the detector to see it at
    conversational range (see SEARCH_CROP), and once locked it tracks, so the
    crop tightens as the person moves away instead of losing them. Each eye
    needs its own: the same face sits at different x in the two views, which is
    the whole basis of the stereo disparity.
    """

    def __init__(self, fw: int, fh: int, search: int = SEARCH_CROP):
        self.fw, self.fh, self.search = fw, fh, search
        self.reset()

    def reset(self) -> None:
        self.side = int(min(self.search, self.fw, self.fh))
        self.x0 = (self.fw - self.side) // 2
        self.y0 = (self.fh - self.side) // 2
        self.locked = False

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y0:self.y0 + self.side, self.x0:self.x0 + self.side]

    def follow(self, px_full: np.ndarray) -> None:
        lo, hi = px_full.min(axis=0), px_full.max(axis=0)
        centre = (lo + hi) / 2.0
        side = int(np.clip(float(np.max(hi - lo)) * ROI_MARGIN,
                           ROI_MIN, min(self.fw, self.fh)))
        self.side = side
        self.x0 = int(np.clip(centre[0] - side / 2, 0, self.fw - side))
        self.y0 = int(np.clip(centre[1] - side / 2, 0, self.fh - side))
        self.locked = True


def detect_face(frame: np.ndarray, roi: Optional[Roi] = None,
                flip_yaw=False, flip_pitch=False) -> Optional[FaceObs]:
    """Detect in `roi`, falling back to a fresh search when the lock is lost."""
    import mediapipe as mp

    def _run(sub: np.ndarray):
        h, w = sub.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(sub, cv2.COLOR_BGR2RGB))
        r = _detector().detect(img)
        return (r, w, h) if r.face_landmarks else (None, w, h)

    if roi is None:
        roi = Roi(frame.shape[1], frame.shape[0])

    res, w, h = _run(roi.crop(frame))
    if res is None and roi.locked:      # lost the lock: widen and try once more
        roi.reset()
        res, w, h = _run(roi.crop(frame))
    if res is None:
        roi.locked = False
        return None

    lm = res.face_landmarks[0]
    px = np.array([[p.x * w + roi.x0, p.y * h + roi.y0] for p in lm], float)
    roi.follow(px)                      # landmarks are full-frame from here on
    blend = ({c.category_name: c.score for c in res.face_blendshapes[0]}
             if res.face_blendshapes else {})

    obs = FaceObs(px=px, blend=blend)
    if res.facial_transformation_matrixes:
        yaw, pitch, roll, fwd = head_angles(res.facial_transformation_matrixes[0])
        obs.yaw = -yaw if flip_yaw else yaw
        obs.pitch = -pitch if flip_pitch else pitch
        obs.roll, obs.fwd = roll, fwd
    return obs


def affect_from_blend(blend: dict) -> tuple[float, float]:
    """Weighted blendshape sums -> (arousal, valence), each roughly in [-1, 1]."""
    out = []
    for axis in ("arousal", "valence"):
        w = BLENDSHAPE_WEIGHTS[axis]
        # normalise by the positive weight mass so the scale does not shift
        # every time a term is added to the table
        norm = sum(abs(v) for v in w.values()) or 1.0
        out.append(float(np.clip(
            sum(blend.get(k, 0.0) * v for k, v in w.items()) / norm * 3.0,
            -1.0, 1.0)))
    return out[0], out[1]


# =============================================================================
# derived channels -- the part with time in it
# =============================================================================
class AffectState:
    """Live 5-state affect label -- the plotter's classify_affect, streamed.

    The classification rule is identical: two independent sign tests on
    baseline-corrected arousal and valence, plus a radius deadband with
    hysteresis for `neutral`. A weighted sum of the two axes would not do, since
    projecting the plane onto a line collides the opposite quadrants -- angry
    and content both land on zero.

    The ONE difference from the offline version, and it is unavoidable: the
    baseline is a running median over the last BASELINE_WINDOW_S rather than the
    median of the whole take, because live code cannot see the future. So the
    label here and the label the plotter assigns to the same recording can
    disagree, most visibly in the first seconds and after a sustained
    expression shifts the running median. Treat the plot as the authority.
    """

    def __init__(self, window_s: float = BASELINE_WINDOW_S):
        self.window_s = window_s
        self.buf: deque[tuple[float, float, float]] = deque()
        self.state = "calibrating"
        self.baseline = (0.0, 0.0)
        self._in_neutral = True

    def update(self, t: float, arousal: float, valence: float,
               have_face: bool) -> str:
        if not have_face:
            # 'unknown', NOT 'neutral'. Neutral is a claim about a face that was
            # looked at; with no face there is nothing to claim. Reporting the
            # two as the same value makes an absence indistinguishable from a
            # calm person, and downstream that is a real bug: a state machine
            # rule reading "Neutral == 1" would fire when someone LEFT.
            # In the FSM's terms this is all five emotion inputs at 0.
            self.state, self._in_neutral = "unknown", True
            return self.state

        self.buf.append((t, arousal, valence))
        while self.buf and t - self.buf[0][0] > self.window_s:
            self.buf.popleft()

        span = self.buf[-1][0] - self.buf[0][0] if len(self.buf) > 1 else 0.0
        if span < BASELINE_MIN_S:
            self.state = "calibrating"      # a median over half a second is noise
            return self.state

        arr = np.asarray(self.buf, float)
        a_base, v_base = float(np.median(arr[:, 1])), float(np.median(arr[:, 2]))
        self.baseline = (a_base, v_base)
        da, dv = arousal - a_base, valence - v_base

        # harder to leave a state than to hold it -- inverting these two makes
        # the deadband easier to escape than to re-enter, which adds chatter
        bar = DEADBAND if self._in_neutral else DEADBAND * DEADBAND_HYST
        if float(np.hypot(da, dv)) < bar:
            self.state, self._in_neutral = "neutral", True
        else:
            self._in_neutral = False
            self.state = STATE_QUADRANT[(da > A_THRESH, dv > V_THRESH)]
        return self.state


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

    def update(self, t: float, pos, obs: Optional[FaceObs]) -> None:
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
    def attending(obs: FaceObs) -> bool:
        return abs(obs.yaw) < ATTEND_YAW_DEG and abs(obs.pitch) < ATTEND_PITCH_DEG

    @property
    def distance(self) -> Optional[float]:
        return None if self.pos is None else float(np.linalg.norm(self.pos))


# =============================================================================
# overlays
# =============================================================================
MESH_C, IRIS_C, RAY_C = (90, 200, 255), (80, 255, 140), (60, 220, 255)


def draw_face(frame: np.ndarray, obs: Optional[FaceObs], eye: str,
              roi: Optional["Roi"] = None) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.putText(out, eye, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    if roi is not None:                       # show what the detector actually sees
        cv2.rectangle(out, (roi.x0, roi.y0), (roi.x0 + roi.side, roi.y0 + roi.side),
                      (90, 200, 120) if roi.locked else (110, 110, 110), 2)
        cv2.putText(out, f"roi {roi.side}px{' lock' if roi.locked else ''}",
                    (roi.x0 + 6, roi.y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (90, 200, 120) if roi.locked else (130, 130, 130), 1)
    if obs is None:
        cv2.putText(out, "no face", (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (70, 70, 220), 1)
        return out

    for x, y in obs.px[:468].astype(int):        # mesh
        cv2.circle(out, (x, y), 1, MESH_C, -1)
    for idx in (IRIS_L, IRIS_R):                 # iris centres
        if idx < len(obs.px):
            cv2.circle(out, tuple(obs.px[idx].astype(int)), 3, IRIS_C, -1)

    # head-pose ray, drawn from the nose in image space
    nose = obs.px[NOSE_TIP].astype(int)
    span = 0.28 * w
    tipx = int(nose[0] + span * np.sin(np.radians(obs.yaw)))
    tipy = int(nose[1] - span * np.sin(np.radians(obs.pitch)))
    cv2.arrowedLine(out, tuple(nose), (tipx, tipy), RAY_C, 2, tipLength=0.25)
    cv2.putText(out, f"yaw {obs.yaw:+5.1f}  pitch {obs.pitch:+5.1f}",
                (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RAY_C, 1)
    return out


WIN = "face probe"


def draw_stop_button(img: np.ndarray) -> tuple[int, int, int, int]:
    """Draw STOP top-right and return its (x, y, w, h) hit box.

    Drawn AFTER any display downscale, so the rectangle is in the same
    coordinate space the mouse callback reports clicks in. Draw it before
    scaling and the hit box lands somewhere else on a resized view.
    """
    h, w = img.shape[:2]
    bw, bh, pad = 96, 32, 12
    x0, y0 = w - bw - pad, pad
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (32, 32, 38), -1)
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (70, 70, 220), 2)
    cv2.putText(img, "STOP", (x0 + 17, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.60, (80, 80, 240), 2, cv2.LINE_AA)
    return x0, y0, bw, bh


def _bar(img, x, y, w, h, frac, color, signed=False, lw=1):
    cv2.rectangle(img, (x, y), (x + w, y + h), (55, 55, 55), lw)
    if signed:
        mid = x + w // 2
        end = int(mid + frac * w / 2)
        cv2.rectangle(img, (min(mid, end), y + lw), (max(mid, end), y + h - lw),
                      color, -1)
        cv2.line(img, (mid, y), (mid, y + h), (110, 110, 110), lw)
    else:
        cv2.rectangle(img, (x + lw, y + lw),
                      (x + lw + int(max(0.0, min(1.0, frac)) * (w - 2 * lw)),
                       y + h - lw),
                      color, -1)


def draw_panel(width: int, ch: Channels, obs: Optional[FaceObs],
               ipd: Optional[float], fps: float, det_ms: float,
               affect: Optional["AffectState"] = None,
               scale: float = 2.0) -> np.ndarray:
    """Readout strip under the video.

    Every dimension derives from `scale` rather than being a literal, because
    the panel is composed at FULL frame width and the whole view is downscaled
    afterwards to fit a screen -- at a 3840-wide capture that is a factor of
    0.42, so anything sized to look right here arrives on screen at less than
    half of it. Scaling the panel up front is what survives that.
    """
    K = scale

    def q(v):
        return int(round(v * K))

    P = np.full((q(190), width, 3), 22, np.uint8)
    f = cv2.FONT_HERSHEY_SIMPLEX
    s = 0.46 * K
    lw = max(1, int(round(K)))
    col = (215, 215, 215)

    def txt(x, y, t, c=col, sc=s):
        cv2.putText(P, t, (q(x), q(y)), f, sc, c, lw, cv2.LINE_AA)

    txt(10, 22, "EXPRESSION CHANNELS", (150, 150, 150), 0.5 * K)

    txt(10, 52, f"arousal {ch.arousal:+.2f}")
    _bar(P, q(130), q(40), q(200), q(14), ch.arousal, (90, 160, 255),
         signed=True, lw=lw)
    txt(10, 78, f"valence {ch.valence:+.2f}")
    _bar(P, q(130), q(66), q(200), q(14), ch.valence, (120, 220, 130),
         signed=True, lw=lw)
    txt(10, 104, f"change  {ch.change:.2f}/s")
    _bar(P, q(130), q(92), q(200), q(14), min(ch.change / 2.0, 1.0),
         (200, 170, 90), lw=lw)

    x2 = 360
    if ch.pos is not None:
        txt(x2, 52, f"head  X{ch.pos[0]:+.2f} Y{ch.pos[1]:+.2f} Z{ch.pos[2]:+.2f} m")
        txt(x2, 78, f"dist  {ch.distance:.2f} m   approach {ch.approach:+.2f} m/s")
    else:
        txt(x2, 52, "head  -- no 3-D fix --", (120, 120, 200))

    attending = obs is not None and Channels.attending(obs)
    txt(x2, 104, f"dwell {ch.dwell:5.1f}s",
        (120, 230, 140) if attending else (140, 140, 140))
    txt(x2 + 130, 104, "ATTENDING" if attending else
        ("absent %.1fs" % ch.absent if obs is None else "looking away"),
        (120, 230, 140) if attending else (140, 140, 140))

    # IPD is the honest check on whether the 3-D numbers mean anything at all
    if ipd is not None:
        ok = 0.050 <= ipd <= 0.080
        txt(10, 140, f"IPD {ipd*1000:5.1f} mm", (120, 230, 140) if ok else (90, 90, 235))
        txt(130, 140,
            "plausible" if ok else "IMPLAUSIBLE - stereo scale is off at face range",
            (120, 230, 140) if ok else (90, 90, 235))
    else:
        txt(10, 140, "IPD --", (130, 130, 130))

    # derived state, drawn large: this is the output the state machine consumes,
    # so it should be readable from across the room rather than squinted at
    if affect is not None:
        st = affect.state
        txt(x2, 140, "state", (150, 150, 150))
        cv2.putText(P, st.upper(), (q(x2 + 62), q(142)), f, 0.72 * K,
                    STATE_BGR.get(st, (200, 200, 200)), lw + 1, cv2.LINE_AA)
        ab, vb = affect.baseline
        txt(x2 + 250, 140, f"base a{ab:+.2f} v{vb:+.2f}", (120, 120, 120), 0.40 * K)

    txt(10, 168, f"{fps:5.1f} fps   detect {det_ms:5.1f} ms", (140, 140, 140))
    txt(x2, 168, "q quit   r reset channels", (140, 140, 140))
    return P


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0",
                    help="camera index, or a video/image path (avoids camera "
                         "contention with the closed loop)")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--flip-yaw", action="store_true")
    ap.add_argument("--flip-pitch", action="store_true")
    ap.add_argument("--config", default=None,
                    help=f"perception YAML (default: {DEFAULT_PERCEPTION_YAML.name})")
    ap.add_argument("--max-width", type=int, default=1600,
                    help="downscale the window to at most this many px wide; "
                         "0 keeps full resolution in a resizable window")
    ap.add_argument("--panel-scale", type=float, default=2.0,
                    help="size multiplier for the readout strip under the video")
    ap.add_argument("--crop", type=int, default=SEARCH_CROP,
                    help="square search-crop side in px; smaller sees further "
                         "but over a narrower field")
    args = ap.parse_args()

    # never get_perception_config(None) -- see DEFAULT_PERCEPTION_YAML
    cfg_path = args.config or (str(DEFAULT_PERCEPTION_YAML)
                               if DEFAULT_PERCEPTION_YAML.exists() else None)
    cfg = get_perception_config(cfg_path)
    if cfg_path is None:
        print("  ! perception YAML not found; using field defaults. "
              "units_to_meters may be wrong and every 3-D value with it.")
    print(f"  config {Path(cfg_path).name if cfg_path else '<defaults>'}: "
          f"units_to_meters={cfg.units_to_meters} "
          f"rotation={cfg.rotation_angle_deg} rotate180={cfg.camera_rotate_180}")
    try:
        calib = load_stereo_calibration(calib_dir=cfg.camera_calibration_path)
        print(f"  stereo calibration loaded from {cfg.camera_calibration_path}")
    except Exception as e:                                   # noqa: BLE001
        calib = None
        print(f"  ! no stereo calibration ({e}); 2-D only, no 3-D head position")

    src: object = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"cannot open source {args.source!r}")
    if isinstance(src, int):
        w, h = cfg.stereo_resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=2)

    ch = Channels()
    affect = AffectState()
    roi_l = roi_r = None
    writer = fh = None
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fh = open(args.csv, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["t", "det_l", "det_r", "x", "y", "z", "dist", "ipd",
                         "yaw", "pitch", "roll", "arousal", "valence", "change",
                         "dwell", "absent", "approach", "attending", "state",
                         "det_ms"])
        print(f"  logging to {args.csv}")

    # Mouse state lives in a dict because the callback runs on the GUI thread
    # and needs somewhere to put the click that the loop can read.
    ui = {"stop": False, "btn": (0, 0, 0, 0)}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        bx, by, bw, bh = ui["btn"]
        if bx <= x <= bx + bw and by <= y <= by + bh:
            ui["stop"] = True

    if not args.headless:
        # AUTOSIZE when scaling: the window is exactly the image, so a click is
        # trivially an image coordinate. NORMAL when not: the full-resolution
        # view is wider than any screen, so the window MUST be resizable to
        # reach the button at all -- and OpenCV maps clicks back into image
        # space for it. That mapping is backend-dependent, so if STOP stops
        # responding after a resize, that is the cause; drop --max-width 0.
        scaling = args.max_width > 0
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE if scaling else cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, on_mouse)
    print("  q or the STOP button quits, r resets the channel state\n")
    t0, frames, fps, last = time.time(), 0, 0.0, time.time()

    try:
        while True:
            tic = time.time()
            ok, frame = read_oriented(cap)
            if not ok:
                print("\n  source ended")
                break
            left, right = split_stereo_frame(frame)

            if roi_l is None:
                roi_l = Roi(left.shape[1], left.shape[0], args.crop)
                roi_r = Roi(right.shape[1], right.shape[0], args.crop)
                print(f"  eye {left.shape[1]}x{left.shape[0]}, "
                      f"search crop {roi_l.side}px "
                      f"(detector scale {192/roi_l.side:.2f} vs "
                      f"{192/left.shape[1]:.2f} uncropped)")

            d0 = time.time()
            fl, fr = pool.map(
                lambda a: detect_face(a[0], a[1], args.flip_yaw, args.flip_pitch),
                ((left, roi_l), (right, roi_r)))
            det_ms = (time.time() - d0) * 1000.0

            # --- stereo fusion: the step MediaPipe does not do for you --------
            pos = ipd = None
            if calib is not None and fl is not None and fr is not None:
                def tri(idx):
                    return triangulate_points(
                        fl.px[idx], fr.px[idx], calib,
                        units_to_m=cfg.units_to_meters,
                        rotation_angle_deg=cfg.rotation_angle_deg,
                        y_translation_m=cfg.y_translation_m,
                        coordinate_limits=FACE_LIMITS)

                pos = tri(NOSE_TIP)
                eye_l, eye_r = tri(IRIS_L), tri(IRIS_R)
                if eye_l is not None and eye_r is not None:
                    ipd = float(np.linalg.norm(eye_l - eye_r))

            obs = fl or fr
            now_t = time.time()
            ch.update(now_t, pos, obs)
            affect.update(now_t - t0, ch.arousal, ch.valence, obs is not None)

            if writer is not None:
                p = ch.pos if ch.pos is not None else (np.nan,) * 3
                writer.writerow([
                    f"{time.time() - t0:.3f}", int(fl is not None), int(fr is not None),
                    *[f"{v:.4f}" for v in p],
                    "" if ch.distance is None else f"{ch.distance:.4f}",
                    "" if ipd is None else f"{ipd:.4f}",
                    *(f"{v:.2f}" for v in ((obs.yaw, obs.pitch, obs.roll)
                                           if obs else (np.nan,) * 3)),
                    f"{ch.arousal:.3f}", f"{ch.valence:.3f}", f"{ch.change:.3f}",
                    f"{ch.dwell:.2f}", f"{ch.absent:.2f}", f"{ch.approach:.3f}",
                    int(obs is not None and Channels.attending(obs)),
                    affect.state, f"{det_ms:.1f}"])

            frames += 1
            if time.time() - last >= 0.5:
                fps = frames / (time.time() - last)
                frames, last = 0, time.time()
                print(f"\r  {'FACE' if obs else '----'} "
                      f"a={ch.arousal:+.2f} v={ch.valence:+.2f} "
                      f"dwell={ch.dwell:4.1f}s "
                      f"dist={'--' if ch.distance is None else f'{ch.distance:.2f}m'} "
                      f"ipd={'--' if ipd is None else f'{ipd*1000:.0f}mm'} "
                      f"{affect.state:11} "
                      f"{fps:4.1f}fps {det_ms:5.1f}ms", end="", flush=True)

            if not args.headless:
                view = np.hstack([draw_face(left, fl, "LEFT", roi_l),
                                  draw_face(right, fr, "RIGHT", roi_r)])
                view = np.vstack([view, draw_panel(view.shape[1], ch, obs,
                                                   ipd, fps, det_ms, affect,
                                                   scale=args.panel_scale)])
                # Two 1920-wide eyes make a 3840-wide view, which no laptop can
                # show at 1:1. Scale to fit, then draw the button on the scaled
                # image so its hit box matches where clicks actually land.
                if args.max_width > 0 and view.shape[1] > args.max_width:
                    s = args.max_width / view.shape[1]
                    view = cv2.resize(view, None, fx=s, fy=s,
                                      interpolation=cv2.INTER_AREA)
                ui["btn"] = draw_stop_button(view)
                cv2.imshow(WIN, view)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q") or ui["stop"]:
                    if ui["stop"]:
                        print("\n  -> STOP clicked")
                    break
                if k == ord("r"):
                    ch = Channels()
                    affect = AffectState()
                    roi_l.reset()
                    roi_r.reset()
                    print("\n  -> channels + affect baseline + ROI reset")

            if isinstance(src, str):                 # play files at ~30 fps
                time.sleep(max(0.0, 1 / 30 - (time.time() - tic)))
    except KeyboardInterrupt:
        pass
    finally:
        pool.shutdown()
        cap.release()
        cv2.destroyAllWindows()
        if fh:
            fh.close()
        print("\n  done.")


if __name__ == "__main__":
    main()
