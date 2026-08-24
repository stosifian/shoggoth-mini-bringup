"""Measure camera tilt (rotation_angle_deg) from a checkerboard of known orientation.

WHY NOT USE THE TENTACLE: its swept trace is only as good as its symmetry, and this
build has a 646-tick tendon spread, ~10 deg residual rest lean and creeping 90A TPU.
The board reconstructs at ~0.5 mm planarity — two orders of magnitude better.

BOARD PLACEMENT — the robot must NOT move (you are measuring the camera's tilt in its
operating pose; tilting the robot measures a pose it will never be in again). Move the
board instead. Two options:

  --orientation vertical  (default)
      Board standing upright, plane perpendicular to the ground. Its COLUMN direction
      then runs along gravity, and that is what we measure. Board must not be rolled
      within its own plane — check a vertical edge with a spirit level.

  --orientation horizontal
      Board lying flat and level, raised on a box until it is in view. Level is a
      property of orientation, not height. Its plane NORMAL is gravity-up.

WHAT IT REPORTS:
  rotation_angle_deg : the value to put in default_perception.yaml
  residual_deg       : the tilt component that rotation-about-X CANNOT remove.
                       The author's transform only rotates about X, so a large
                       residual means your camera has roll/yaw the 2-parameter model
                       cannot express, and you need a full rigid transform.

  python tools/fit_camera_tilt.py --camera 2
  python tools/fit_camera_tilt.py --camera 2 --orientation horizontal
  python tools/fit_camera_tilt.py --camera 2 --frames 30 --cols 9 --rows 6
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.configs import get_perception_config  # noqa: E402
from shoggoth_mini.perception.camera import read_oriented  # noqa: E402
from shoggoth_mini.perception.stereo import (  # noqa: E402
    load_stereo_calibration,
    split_stereo_frame,
    undistort_points,
)

DEFAULT_CFG = "shoggoth_mini/configs/default_perception.yaml"


def find_corners(gray, pattern):
    ok, c = cv2.findChessboardCornersSB(gray, pattern)
    if ok:
        return c
    ok, c = cv2.findChessboardCorners(
        gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    return c if ok else None


def triangulate_grid(frame, calib, pattern, diagnose=False):
    """Return (rows, cols, 3) board corners in the SAME frame the transform sees.

    With diagnose=True returns (grid, reason) so the caller can report WHICH half
    failed — the difference between "board out of stereo overlap" and "wrong
    --cols/--rows" is otherwise invisible.
    """
    left, right = split_stereo_frame(frame)
    gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    cl, cr = find_corners(gl, pattern), find_corners(gr, pattern)
    if cl is None or cr is None:
        if diagnose:
            if cl is None and cr is None:
                return None, "both"
            return None, ("left" if cl is None else "right")
        return None

    K1, D1, P1, K2, D2, P2 = calib.as_tuple()
    ul = undistort_points(cl.reshape(-1, 2), K1, D1, P1, calib.R1)
    ur = undistort_points(cr.reshape(-1, 2), K2, D2, P2, calib.R2)
    pts4 = cv2.triangulatePoints(P1[:3], P2[:3],
                                 ul.T.astype(np.float64), ur.T.astype(np.float64))
    pts3 = (pts4[:3] / pts4[3]).T

    # Match stereo.py's sign convention (+Y up, forward = -Z) so the angle we solve
    # for is directly the one triangulate_points() will apply.
    pts3[:, 1] *= -1.0
    pts3[:, 2] *= -1.0

    cols, rows = pattern
    grid = pts3.reshape(rows, cols, 3)
    return (grid, None) if diagnose else grid


def up_vector(grid, orientation):
    """Unit vector along gravity-up, in camera coordinates."""
    if orientation == "vertical":
        # Column direction: consecutive ROWS differ along the board's vertical axis.
        v = np.diff(grid, axis=0).reshape(-1, 3).mean(axis=0)
    else:
        # Horizontal board: plane normal via SVD.
        pts = grid.reshape(-1, 3)
        v = np.linalg.svd(pts - pts.mean(axis=0))[2][-1]
    v = v / np.linalg.norm(v)
    return v if v[1] >= 0 else -v  # disambiguate sign: "up" has +Y


def solve_x_rotation(u):
    """Angle about X that best maps u onto +Y, plus the unremovable residual.

    stereo.py rotates about X only:
        y' = y cos a - z sin a ;  z' = y sin a + z cos a  ;  x unchanged
    Driving z' to zero gives a = atan2(-uz, uy). The X component survives untouched,
    so it becomes the residual.
    """
    ux, uy, uz = u
    a = np.arctan2(-uz, uy)
    ca, sa = np.cos(a), np.sin(a)
    rotated = np.array([ux, uy * ca - uz * sa, uy * sa + uz * ca])
    rotated /= np.linalg.norm(rotated)
    residual = np.degrees(np.arccos(np.clip(rotated @ np.array([0.0, 1.0, 0.0]), -1, 1)))
    return np.degrees(a), residual, rotated


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--camera", type=int, default=None, help="index (default: config)")
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--cols", type=int, default=9, help="INTERNAL corners across")
    ap.add_argument("--rows", type=int, default=6, help="INTERNAL corners down")
    ap.add_argument("--orientation", choices=["vertical", "horizontal"],
                    default="vertical")
    ap.add_argument("--frames", type=int, default=20, help="frames to average")
    ap.add_argument("--base-height", action="store_true",
                    help="measure y_translation_m instead of tilt: hold the board so "
                         "its surface is LEVEL WITH THE TENTACLE BASE (where the "
                         "tentacle leaves the dome), and this reports the base's "
                         "height in the rotated camera frame. Avoids rulering to the "
                         "optical centre, which is buried inside the dome.")
    ap.add_argument("--tilt-deg", type=float, default=None,
                    help="rotation_angle_deg to apply for --base-height "
                         "(default: value from config)")
    args = ap.parse_args()

    cfg = get_perception_config(args.config)
    idx = args.camera if args.camera is not None else cfg.camera_index
    pattern = (args.cols, args.rows)

    calib = load_stereo_calibration(calib_dir=Path(cfg.camera_calibration_path))
    if not calib.has_rectification():
        print("WARNING: calibration has no R1/R2 — recalibrate, results will be biased.")

    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.stereo_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.stereo_resolution[1])
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {idx}")

    print(f"\ncamera {idx}, board {args.cols}x{args.rows}, {args.orientation} board")
    print(f"collecting {args.frames} detections — hold the board still...\n")

    ups, planarity, grids = [], [], []
    tries, fails_l, fails_r, fails_both = 0, 0, 0, 0
    t_start = time.time()
    while len(ups) < args.frames and tries < args.frames * 20:
        tries += 1
        ok, frame = read_oriented(cap)
        if not ok:
            print("\n  frame grab failed — camera disconnected?")
            break

        # Report on EVERY attempt: a silent failing loop looks identical to a hang.
        grid, why = triangulate_grid(frame, calib, pattern, diagnose=True)
        if grid is None:
            if why == "left":
                fails_l += 1
            elif why == "right":
                fails_r += 1
            else:
                fails_both += 1
        else:
            ups.append(up_vector(grid, args.orientation))
            grids.append(grid)
            pts = grid.reshape(-1, 3)
            n = np.linalg.svd(pts - pts.mean(axis=0))[2][-1]
            planarity.append(
                float(np.sqrt(np.mean(((pts - pts.mean(axis=0)) @ n) ** 2)))
            )

        rate = tries / max(time.time() - t_start, 1e-6)
        print(f"  got {len(ups):3d}/{args.frames}  |  attempts {tries:4d} "
              f"({rate:.1f}/s)  miss: both={fails_both} left-only={fails_r} "
              f"right-only={fails_l}   ", end="\r", flush=True)
    cap.release()
    print()

    if fails_both or fails_l or fails_r:
        print(f"\n  detection misses — both halves:{fails_both} "
              f"left-half-only:{fails_r} right-half-only:{fails_l}")
        if fails_l and not fails_r:
            print("  Board is missing from the RIGHT half — move it toward the "
                  "stereo overlap.")
        elif fails_r and not fails_l:
            print("  Board is missing from the LEFT half — move it toward the "
                  "stereo overlap.")
        elif fails_both:
            print("  Not detected in EITHER half: check --cols/--rows (INTERNAL "
                  "corners), lighting/glare, and that the whole board is in view.")

    if len(ups) < 3:
        raise SystemExit(f"\nonly {len(ups)} detections — check board visible in BOTH "
                         f"halves, lighting, and --cols/--rows")

    if args.base_height:
        tilt = args.tilt_deg if args.tilt_deg is not None else cfg.rotation_angle_deg
        a = np.radians(tilt)
        ca, sa = np.cos(a), np.sin(a)
        ys = []
        for g in grids:
            pts = g.reshape(-1, 3)
            # same rotation-about-X that triangulate_points() applies
            y_rot = pts[:, 1] * ca - pts[:, 2] * sa
            ys.append(float(np.mean(y_rot)))
        ys = np.array(ys)
        print(f"\ndetections used     : {len(ys)}")
        print(f"planarity RMS       : {np.mean(planarity)*1000:.2f} mm")
        print(f"applied tilt        : {tilt:.2f} deg")
        print(f"board Y (rotated)   : {ys.mean():+.4f} m  (std {ys.std():.4f})")
        print("\n" + "=" * 58)
        print(f"  y_translation_m : {-ys.mean():+.4f}")
        print("=" * 58)
        print("\n  (negated: the offset SUBTRACTS the base height so the base sits")
        print("   at Y=0. Sanity check — upstream used -0.03, i.e. a base ~3 cm")
        print("   above the camera. A wildly different magnitude means the board")
        print("   was not level with where the tentacle leaves the dome.)")
        return 0

    U = np.array(ups)
    u = U.mean(axis=0)
    u /= np.linalg.norm(u)
    spread = np.degrees(np.arccos(np.clip(U @ u, -1, 1)))

    angle, residual, rotated = solve_x_rotation(u)

    print(f"\n\ndetections used     : {len(ups)}")
    print(f"planarity RMS       : {np.mean(planarity)*1000:.2f} mm")
    print(f"frame-to-frame spread: {spread.mean():.2f} deg (max {spread.max():.2f})")
    print(f"measured up vector  : [{u[0]:+.4f} {u[1]:+.4f} {u[2]:+.4f}]")
    print("\n" + "=" * 58)
    print(f"  rotation_angle_deg : {angle:+.2f}")
    print(f"  residual tilt      : {residual:.2f} deg  <- rotation-about-X cannot fix")
    print("=" * 58)

    if residual < 2.0:
        print("\n  Residual small: the author's 2-parameter transform fits your mount.")
    else:
        print(f"\n  Residual {residual:.1f} deg is LARGE. Your camera has roll/yaw that")
        print("  rotation-about-X cannot express — a full rigid transform is needed,")
        print("  or the board was rolled in its own plane (re-check with a level).")
    if spread.mean() > 1.0:
        print(f"  NOTE: {spread.mean():.1f} deg frame-to-frame spread is high — board or")
        print("  camera moving, or detections marginal. Steady the board and re-run.")

    print("\nNext: set rotation_angle_deg in default_perception.yaml, then measure")
    print("y_translation_m with a ruler (base height above the camera; the base is")
    print("hidden behind the dome so it cannot be triangulated).")


if __name__ == "__main__":
    raise SystemExit(main())
