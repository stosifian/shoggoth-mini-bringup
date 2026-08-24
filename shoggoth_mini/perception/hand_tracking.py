"""Hand tracking using MediaPipe for gesture recognition and 3D positioning.

Ported from the removed legacy `mp.solutions.hands` API to the MediaPipe Tasks
API (HandLandmarker). The public interface is unchanged: ``get_mediapipe_hand_data``
still returns (index_tip_pixel_xy, results), and ``results`` still exposes
``.multi_hand_landmarks[i].landmark[k].x/.y/.z`` so existing consumers
(orchestrator, closed_loop, dashboard, debug_perception) need no changes.
"""

import logging
import threading
from pathlib import Path
from typing import Optional, Tuple, Any, List
from collections import deque

import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Tasks API import is guarded so the module stays importable (keeping the whole
# CLI usable for RL/vision) even on an install without mediapipe's tasks extras.
try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )

    _TASKS_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on install
    _TASKS_AVAILABLE = False
    _IMPORT_ERROR = exc
    logger.warning("mediapipe Tasks API unavailable (%s); hand tracking disabled.", exc)

# Model bundle (downloaded once into the repo). Override with $MEDIAPIPE_HAND_MODEL.
import os

_DEFAULT_MODEL = (
    Path(__file__).resolve().parents[2] / "assets/models/vision/hand_landmarker.task"
)
MODEL_PATH = Path(os.environ.get("MEDIAPIPE_HAND_MODEL", str(_DEFAULT_MODEL)))

# Thread-local storage for detector instances (Tasks detectors aren't thread-safe)
_thread_local = threading.local()

# Hand landmark indices
INDEX_FINGER_TIP = 8
INDEX_FINGER_MCP = 5
THUMB_TIP = 4
MIDDLE_FINGER_TIP = 12
RING_FINGER_TIP = 16
PINKY_TIP = 20

# Standard MediaPipe 21-landmark hand skeleton (for drawing)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                # palm base
]


class _Hand:
    """Compat shim: mimics legacy hand_landmarks with a ``.landmark`` list.

    Elements are Tasks NormalizedLandmark objects, which already expose
    ``.x``, ``.y``, ``.z`` in normalized [0, 1] image coordinates.
    """

    def __init__(self, landmarks: List[Any]):
        self.landmark = landmarks


class _Results:
    """Compat shim exposing ``.multi_hand_landmarks`` like the legacy API."""

    def __init__(self, hand_landmarks_list: List[List[Any]]):
        self.multi_hand_landmarks = (
            [_Hand(lms) for lms in hand_landmarks_list] if hand_landmarks_list else None
        )


def _get_thread_hands_detector() -> "HandLandmarker":
    """Get (or lazily create) the thread-local HandLandmarker detector."""
    if not _TASKS_AVAILABLE:
        raise RuntimeError(
            f"mediapipe Tasks API is unavailable: {_IMPORT_ERROR}"
        )
    if getattr(_thread_local, "hands_detector", None) is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Hand landmarker model not found at {MODEL_PATH}. Download it "
                "from https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task or set "
                "$MEDIAPIPE_HAND_MODEL."
            )
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.25,
        )
        _thread_local.hands_detector = HandLandmarker.create_from_options(options)
    return _thread_local.hands_detector


def get_mediapipe_hand_data(
    frame: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[Any]]:
    """Detect index finger tip using MediaPipe (convenience function).

    Args:
        frame: Input image as BGR numpy array

    Returns:
        Tuple of (index_finger_tip_position_xy_pixels, results_compat_object)
    """
    hands_detector = _get_thread_hands_detector()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    detection = hands_detector.detect(mp_image)
    results = _Results(detection.hand_landmarks)

    index_finger_tip_pos = None
    if results.multi_hand_landmarks:
        tip_landmark = results.multi_hand_landmarks[0].landmark[INDEX_FINGER_TIP]
        h, w = frame.shape[:2]
        index_finger_tip_pos = np.array(
            [tip_landmark.x * w, tip_landmark.y * h], dtype=np.float32
        )

    return index_finger_tip_pos, results


def draw_hand_landmarks(
    image: np.ndarray,
    results: Any,
    connections_color: Tuple[int, int, int] = (0, 255, 0),
    landmarks_color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Draw hand landmarks and connections on image.

    Args:
        image: Input image to draw on
        results: results object from get_mediapipe_hand_data
        connections_color: Color for hand connections (BGR)
        landmarks_color: Color for landmarks (BGR)

    Returns:
        Image with drawn landmarks
    """
    annotated_image = image.copy()
    if not results or not results.multi_hand_landmarks:
        return annotated_image

    h, w = annotated_image.shape[:2]
    for hand in results.multi_hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmark]
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(annotated_image, pts[a], pts[b], connections_color, 2)
        for p in pts:
            cv2.circle(annotated_image, p, 3, landmarks_color, -1)

    return annotated_image


def is_wave_gesture(
    x_coordinates: deque,
    amplitude_threshold: float = 0.2,
    min_coordinates: int = 10,
    min_velocity_points: int = 10,
    min_zero_crossings: int = 5,
) -> Tuple[bool, Optional[np.ndarray]]:
    """Detect wave gesture from x-coordinate trail.

    Args:
        x_coordinates: Deque of x-coordinate values
        amplitude_threshold: Minimum amplitude for wave detection
        min_coordinates: Minimum number of coordinate points needed
        min_velocity_points: Minimum velocity points for analysis
        min_zero_crossings: Minimum zero crossings for wave detection

    Returns:
        Tuple of (is_wave_detected, velocity_array)
    """
    if len(x_coordinates) < min_coordinates:
        return False, None

    x_array = np.array(list(x_coordinates))
    velocity = np.diff(x_array)

    if len(velocity) < min_velocity_points:
        return False, None

    # Find zero crossings in velocity (direction changes)
    zero_crossings_indices = np.where(
        np.sign(velocity[:-1]) * np.sign(velocity[1:]) < 0
    )[0]
    num_zero_crossings = len(zero_crossings_indices)

    # Calculate amplitude
    amplitude = x_array.max() - x_array.min()

    # Detect wave
    wave_detected = (
        num_zero_crossings >= min_zero_crossings and amplitude > amplitude_threshold
    )

    return wave_detected, velocity if wave_detected else None


def update_landmark_trail(
    current_trail: deque,
    mediapipe_results: Any,
    landmark_id: int,
    last_seen_time: Optional[float],
    current_time: float,
    timeout_duration: float = 2.0,
) -> Tuple[Optional[float], bool, Optional[float]]:
    """Update landmark trail with timeout management.

    Args:
        current_trail: Deque to store landmark coordinates
        mediapipe_results: results object from get_mediapipe_hand_data
        landmark_id: Landmark index to track
        last_seen_time: Last time landmark was detected
        current_time: Current timestamp
        timeout_duration: Time after which to clear trail if no detection

    Returns:
        Tuple of (updated_last_seen_time, trail_was_cleared, extracted_coordinate)
    """
    trail_cleared = False
    extracted_coord = None

    if mediapipe_results and mediapipe_results.multi_hand_landmarks:
        hand_landmarks = mediapipe_results.multi_hand_landmarks[0]
        try:
            extracted_coord = hand_landmarks.landmark[landmark_id].x
            current_trail.append(extracted_coord)
            updated_last_seen_time = current_time
        except IndexError:
            logger.warning(f"Warning: Landmark index {landmark_id} is invalid.")
            updated_last_seen_time = last_seen_time
    else:
        updated_last_seen_time = last_seen_time
        if last_seen_time is not None and (
            current_time - last_seen_time > timeout_duration
        ):
            if current_trail:
                current_trail.clear()
                trail_cleared = True
            updated_last_seen_time = None

    return updated_last_seen_time, trail_cleared, extracted_coord


def close_mediapipe_hands():
    """Close the thread-local MediaPipe hands detector instance."""
    detector = getattr(_thread_local, "hands_detector", None)
    if detector is not None:
        detector.close()
        _thread_local.hands_detector = None
