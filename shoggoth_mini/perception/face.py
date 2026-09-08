"""Stereo face landmarks: find a face, and keep enough pixels on it to see one.

Sibling of hand_tracking.py -- same MediaPipe Tasks API, same thread-local
detector, same reason for it: the detector is not thread safe and the two eyes
are processed concurrently.

This is PERCEPTION, not interpretation. It answers "where is the face and what
is it doing", and hands the answer to affect/ to decide what that means. It
lives in the package rather than in a tool because the orchestrator needs it as
much as the debugging viewer does.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..affect import head_angles

PKG = Path(__file__).resolve().parents[1].parent
MODEL_PATH = PKG / "assets" / "models" / "vision" / "face_landmarker.task"

# MediaPipe's face detector runs on a ~192 px square. A full 1920x760 eye is
# therefore scaled by 0.1, so a 150 mm face at 0.5 m (186 px at fx=620) reaches
# the detector as ~19 px -- right at BlazeFace's floor, which is why detection
# dies just past half a metre. Cropping to a square recovers the pixels: 760 px
# gives 0.25 scale and ~47 px on the same face.
SEARCH_CROP = 760
ROI_MIN = 160
ROI_MARGIN = 2.2            # crop side as a multiple of the face's larger extent

# The crop is an INPUT to a deterministic but extremely sensitive regressor:
# one least-significant bit of pixel noise already moves pitch by ~0.2 deg, so
# reframing by a pixel is not a small perturbation. Measured with
# tools/jitter_probe.py, the box was stepping ~1 px EVERY frame on a subject
# holding still, and that accounted for ~61% of live yaw variance (reproduced
# across two sittings; the pitch share was not reproducible, 7% and 35%).
#
# The cause is NOT the centre, which is steady to 0.23 px on a real mesh. It is
# the SIDE: extent jitters 0.43% per frame, ROI_MARGIN multiplies that by 2.2
# into 2.08 px of side, and x0 = centre - side/2 inherits half of it. So the
# stabilisation belongs on the size, and quantising it is enough -- the box
# only resizes when the face genuinely changes scale.
#
# Both rules are deadzones rather than filters, deliberately. An EMA still
# nudges the box every frame, which is the thing being removed; a deadzone
# holds it exactly still and then tracks with no lag at all once the face has
# really moved. Replayed against 896 recorded frames of a real face: the box is
# unchanged on 99% of frames (was 23%), while a 500 px slide still tracks with
# 0.2 px median error and a 1.7x approach resizes to within 0%.
ROI_SIDE_STEP = 16          # side snaps to this grid, in px
ROI_SIDE_HYST = 0.75        # ...and only when it is off by this much of a step
ROI_CENTRE_DEADZONE = 3.0   # centre moves under this px leave the box alone

# 478-landmark layout: 0..467 mesh, 468..472 left iris, 473..477 right iris.
# "Left" is MediaPipe's naming (the subject's left), not the image side.
NOSE_TIP = 1
IRIS_L, IRIS_R = 468, 473

_thread_local = threading.local()


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
        # Float tracking state, held separately from the rounded x0/y0/side the
        # crop uses. Rounding the running value instead would put a sub-pixel
        # step back in every frame, which is most of what this removes.
        self._c: Optional[np.ndarray] = None
        self._s: float = float(self.side)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y0:self.y0 + self.side, self.x0:self.x0 + self.side]

    def follow(self, px_full: np.ndarray) -> None:
        """Track the face, but hold the box completely still when it is still.

        See the ROI_SIDE_STEP block above for why the size, not the centre, is
        what needed stabilising.
        """
        lo, hi = px_full.min(axis=0), px_full.max(axis=0)
        want_c = (lo + hi) / 2.0
        want_s = float(np.clip(float(np.max(hi - lo)) * ROI_MARGIN,
                               ROI_MIN, min(self.fw, self.fh)))

        if not self.locked or self._c is None:
            c, s = want_c.astype(float), want_s          # first lock: snap
        else:
            c, s = self._c, self._s
            if abs(want_s - s) > ROI_SIDE_STEP * ROI_SIDE_HYST:
                s = round(want_s / ROI_SIDE_STEP) * ROI_SIDE_STEP
            if float(np.hypot(*(want_c - c))) > ROI_CENTRE_DEADZONE:
                c = want_c.astype(float)

        self._c, self._s = c, s
        side = int(np.clip(round(s), ROI_MIN, min(self.fw, self.fh)))
        self.side = side
        self.x0 = int(np.clip(round(c[0] - side / 2), 0, self.fw - side))
        self.y0 = int(np.clip(round(c[1] - side / 2), 0, self.fh - side))
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
