"""Stereo vision utilities for 3D triangulation and camera geometry."""

import pickle
from pathlib import Path
from typing import Optional, Tuple, Any, Dict
import numpy as np
import cv2
from dataclasses import dataclass

from ..configs.loaders import get_perception_config


def load_stereo_calibration(
    pair: str = "camera-1-camera-2", calib_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Load stereo calibration parameters from pickle file.

    Args:
        pair: Camera pair identifier
        calib_dir: Directory containing calibration data. If None, uses config.

    Returns:
        Dictionary containing calibration matrices and parameters

    Raises:
        FileNotFoundError: If calibration file not found
        KeyError: If camera pair not found in calibration data
    """
    if calib_dir is None:
        config = get_perception_config()
        calib_dir = config.camera_calibration_path

    calib_path = calib_dir / "stereo_params.pickle"
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    with open(calib_path, "rb") as fh:
        stereo_data = pickle.load(fh)

    if pair not in stereo_data:
        raise KeyError(f"Camera pair '{pair}' not found in calibration data")

    return StereoCalibration.from_raw(stereo_data[pair])


def undistort_points(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    projection_matrix: np.ndarray,
    rectification_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Undistort 2D points using camera calibration parameters.

    Args:
        points: Input points as (N, 2) array
        camera_matrix: Camera intrinsic matrix K
        distortion_coeffs: Distortion coefficients D
        projection_matrix: Projection matrix P
        rectification_matrix: Rectification rotation R (R1 for left, R2 for right)
            from cv2.stereoRectify. REQUIRED when P came from stereoRectify, since
            P is expressed in the rectified frame: without R the points stay in the
            original camera frame and triangulation carries a systematic scale
            error that no amount of re-measuring the board can remove. Pass None
            only for calibrations whose P matrices are already unrectified.

    Returns:
        Undistorted points in projection matrix reference frame
    """
    pts = np.asarray(points, np.float32).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(
        pts,
        camera_matrix,
        distortion_coeffs,
        R=rectification_matrix,
        P=projection_matrix,
    )
    return undistorted.reshape(points.shape)


@dataclass
class StereoCalibration:
    """Standardized stereo calibration container.

    Attributes map directly to the symbols typically used in stereo geometry:
    * ``K1, D1, P1`` – intrinsics, distortion, projection for left camera.
    * ``K2, D2, P2`` – same for right camera.
    """

    K1: np.ndarray
    D1: np.ndarray
    P1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    P2: np.ndarray
    # Rectification rotations from cv2.stereoRectify. Optional for backward
    # compatibility with older pickles that predate them; when absent, points are
    # undistorted into the original camera frame (see undistort_points).
    R1: Optional[np.ndarray] = None
    R2: Optional[np.ndarray] = None

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "StereoCalibration":
        """Create :class:`StereoCalibration` from raw dict."""

        return cls(
            K1=raw["cameraMatrix1"],
            D1=raw["distCoeffs1"],
            P1=raw["P1"],
            K2=raw["cameraMatrix2"],
            D2=raw["distCoeffs2"],
            P2=raw["P2"],
            R1=raw.get("R1"),
            R2=raw.get("R2"),
        )

    def as_tuple(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the six OpenCV matrices as a tuple (K1, D1, P1, K2, D2, P2)."""

        return self.K1, self.D1, self.P1, self.K2, self.D2, self.P2

    def has_rectification(self) -> bool:
        """True when both rectification rotations are available."""

        return self.R1 is not None and self.R2 is not None


def triangulate_points(
    xy_left: Optional[np.ndarray],
    xy_right: Optional[np.ndarray],
    calib: "StereoCalibration",
    *,
    units_to_m: float = 0.05,
    rotation_angle_deg: float = 35,
    y_translation_m: float = -0.03,
    coordinate_limits: Optional[Dict[str, Dict[str, float]]] = None,
) -> Optional[np.ndarray]:
    """Triangulate 3-D point from two 2-D correspondences.

    Args:
        xy_left: 2-D point in left frame (or None)
        xy_right: 2-D point in right frame (or None)
        calib: Stereo calibration parameters as :class:`StereoCalibration`
        units_to_m: Scale factor to convert to metres
        rotation_angle_deg: Rotation (deg) around X-axis for alignment
        y_translation_m: Y translation offset (m)
        coordinate_limits: Optional clipping bounds

    Returns
    -------
    Optional[np.ndarray]
        3-D point
    """

    if xy_left is None or xy_right is None:
        return None

    K1, D1, P1, K2, D2, P2 = calib.as_tuple()

    # Undistort into the frame P1/P2 are expressed in. When the calibration carries
    # rectification rotations, P1/P2 came from stereoRectify and R1/R2 MUST be
    # applied — otherwise points stay in the original camera frame and every
    # triangulated position picks up a systematic scale error.
    undistorted_left = undistort_points(xy_left, K1, D1, P1, calib.R1)
    undistorted_right = undistort_points(xy_right, K2, D2, P2, calib.R2)

    # Triangulate homogeneous point then convert to Euclidean
    point_4d = cv2.triangulatePoints(
        P1[:3], P2[:3], undistorted_left.reshape(2, 1), undistorted_right.reshape(2, 1)
    )
    point_3d = (point_4d[:3] / point_4d[3]).ravel()

    point_3d[0] = point_3d[0] * units_to_m
    point_3d[1] = -point_3d[1] * units_to_m
    point_3d[2] = -point_3d[2] * units_to_m

    # Rotation about X
    angle_rad = np.radians(rotation_angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    y_old, z_old = point_3d[1], point_3d[2]
    point_3d[1] = y_old * cos_a - z_old * sin_a
    point_3d[2] = y_old * sin_a + z_old * cos_a

    # Translation
    point_3d[1] += y_translation_m

    point_3d[0] = np.clip(
        point_3d[0],
        coordinate_limits["X"]["clip_min"],
        coordinate_limits["X"]["clip_max"],
    )
    point_3d[1] = np.clip(
        point_3d[1],
        coordinate_limits["Y"]["clip_min"],
        coordinate_limits["Y"]["clip_max"],
    )
    point_3d[2] = np.clip(
        point_3d[2],
        coordinate_limits["Z"]["clip_min"],
        coordinate_limits["Z"]["clip_max"],
    )

    return point_3d


def split_stereo_frame(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split side-by-side stereo frame into left and right frames.

    Args:
        frame: Stereo frame with left and right images side by side

    Returns:
        Tuple of (left_frame, right_frame)
    """
    half_width = frame.shape[1] // 2
    left_frame = frame[:, :half_width]
    right_frame = frame[:, half_width:]
    return left_frame, right_frame


__all__ = [
    "StereoCalibration",
    "load_stereo_calibration",
    "triangulate_points",
    "split_stereo_frame",
]
