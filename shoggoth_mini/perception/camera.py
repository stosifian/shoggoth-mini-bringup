"""Single funnel for camera frame acquisition and orientation.

Every capture site in the project reads frames through here so that orientation is
applied in exactly one place. That matters more than it looks: a 180 degree camera
rotation SWAPS which half of a side-by-side stereo frame belongs to which physical
camera. `split_stereo_frame()` slices by pixel position, so an inverted board makes
it assign left and right backwards, which inverts the stereo baseline and produces
triangulation that is geometrically wrong while still looking plausible.

The failure mode this guards against is divergence: a rotation applied during
calibration but not at runtime (or vice versa) silently invalidates every 3D
estimate, with no error anywhere.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_orientation_logged = False
_cached_rotate_180: Optional[bool] = None

# The project's default perception YAML. This must be passed explicitly:
# get_perception_config() with NO argument returns pydantic field defaults and
# silently ignores the YAML, which would leave the orientation fix permanently off.
_DEFAULT_PERCEPTION_YAML = (
    Path(__file__).resolve().parent.parent / "configs" / "default_perception.yaml"
)


def _resolve_rotate_180() -> bool:
    """Read camera_rotate_180 from the default perception YAML, cached."""
    global _cached_rotate_180
    if _cached_rotate_180 is not None:
        return _cached_rotate_180

    value = False
    try:
        from ..configs import get_perception_config

        if _DEFAULT_PERCEPTION_YAML.exists():
            cfg = get_perception_config(str(_DEFAULT_PERCEPTION_YAML))
        else:
            logger.warning(
                "Perception YAML not found at %s; camera orientation defaults to OFF",
                _DEFAULT_PERCEPTION_YAML,
            )
            cfg = get_perception_config()
        value = bool(cfg.camera_rotate_180)
    except Exception as e:  # config unavailable (standalone tools, tests)
        logger.debug("Could not read camera_rotate_180 from config: %s", e)

    _cached_rotate_180 = value
    return value


def set_camera_orientation(rotate_180: bool) -> None:
    """Override the configured orientation for the life of the process.

    For callers that have already loaded their own perception config and want its
    value to win over the default YAML.
    """
    global _cached_rotate_180, _orientation_logged
    _cached_rotate_180 = bool(rotate_180)
    _orientation_logged = False


def apply_camera_orientation(
    frame: np.ndarray, rotate_180: Optional[bool] = None
) -> np.ndarray:
    """Apply the configured orientation fix to a freshly captured frame.

    Args:
        frame: Raw frame straight from VideoCapture.read().
        rotate_180: Override the configured value. When None, the perception
            config's `camera_rotate_180` is used.

    Returns:
        The oriented frame — rotated 180 degrees if configured, else unchanged.
    """
    global _orientation_logged

    if frame is None:
        return frame

    if rotate_180 is None:
        rotate_180 = _resolve_rotate_180()

    if not rotate_180:
        return frame

    if not _orientation_logged:
        logger.info("Camera orientation: rotating frames 180 degrees at capture")
        _orientation_logged = True

    return cv2.rotate(frame, cv2.ROTATE_180)


def read_oriented(
    cap: cv2.VideoCapture, rotate_180: Optional[bool] = None
) -> Tuple[bool, Optional[np.ndarray]]:
    """Drop-in replacement for `cap.read()` that applies the orientation fix.

    Returns:
        (ok, frame) exactly like cv2.VideoCapture.read(), with the frame oriented.
    """
    ok, frame = cap.read()
    if not ok:
        return ok, frame
    return ok, apply_camera_orientation(frame, rotate_180=rotate_180)


__all__ = ["apply_camera_orientation", "read_oriented"]
