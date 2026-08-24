"""Stereo-calibrate the ELP camera from captured checkerboard pairs — OpenCV only.

Replaces the DeepLabCut notebook route (DLC isn't installed and is painful on
arm64). Consumes the image pairs written by `record stereo-calibration`
(camera-1-NN.jpg / camera-2-NN.jpg) and writes the SAME pickle the runtime loads:

    assets/hardware/calibration/stereo_params.pickle
      -> { "camera-1-camera-2": {cameraMatrix1, distCoeffs1, P1,
                                 cameraMatrix2, distCoeffs2, P2} }

(plus camera-1/2_intrinsic_params.pickle for parity with the shipped set).
These keys/matrices are exactly what shoggoth_mini/perception/stereo.py reads.

CORNER COUNT: pass INTERNAL corners = (squares - 1) per side.
A board of 10x7 SQUARES has 9x6 internal corners -> --cols 9 --rows 6.

SQUARE SIZE: measure a square on the tablet with a ruler and pass --square-mm.
Absolute scale is later rescaled by units_to_meters in default_perception.yaml,
so ballpark is fine, but get it roughly right.

Run from repo root, venv active:

  python tools/stereo_calibrate.py --dir data/calibration --cols 9 --rows 6 --square-mm 18

Detection overlays are saved to <dir>/corners/ — EYEBALL THEM: keep only pairs
where every corner is found in the right order; re-run after deleting bad pairs.
"""
import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

# allow importing the shoggoth_mini package when run as a script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def verify_calibration(image_dir, pattern, square_mm, calib_dir):
    """Triangulate detected board corners with the saved calibration and check
    metric scale (reconstructed square size vs input) + planarity residual.

    Uses the SAME undistort/triangulate path as the runtime (stereo.py), minus
    the empirical robot-frame transform, so points come out in metric camera
    coordinates. Reconstructed neighbour spacing should equal the square size,
    and all corners should lie on a plane (the board is flat).
    """
    from shoggoth_mini.perception.stereo import undistort_points, StereoCalibration

    calib_path = calib_dir / "stereo_params.pickle"
    if not calib_path.exists():
        raise SystemExit(f"no calibration at {calib_path} — run without --verify first")
    with open(calib_path, "rb") as fh:
        calib = StereoCalibration.from_raw(pickle.load(fh)["camera-1-camera-2"])
    K1, D1, P1, K2, D2, P2 = calib.as_tuple()

    cols, rows = pattern
    spacings_mm, planarity_mm = [], []
    used = 0
    for lp in sorted(image_dir.glob("camera-1-*.jpg")):
        rp = image_dir / lp.name.replace("camera-1-", "camera-2-")
        if not rp.exists():
            continue
        gl = cv2.cvtColor(cv2.imread(str(lp)), cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(cv2.imread(str(rp)), cv2.COLOR_BGR2GRAY)
        cl, cr = find_corners(gl, pattern), find_corners(gr, pattern)
        if cl is None or cr is None:
            continue
        used += 1

        # runtime path: undistort into rectified projective frame, then DLT
        ul = undistort_points(cl.reshape(-1, 2), K1, D1, P1, calib.R1)
        ur = undistort_points(cr.reshape(-1, 2), K2, D2, P2, calib.R2)
        pts4 = cv2.triangulatePoints(P1[:3], P2[:3],
                                     ul.T.astype(np.float64), ur.T.astype(np.float64))
        pts3 = (pts4[:3] / pts4[3]).T  # (N, 3) in metres

        # neighbour spacings along rows and columns (should equal one square)
        grid = pts3.reshape(rows, cols, 3)
        d_h = np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel()
        d_v = np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel()
        spacings_mm.extend((np.concatenate([d_h, d_v]) * 1000.0).tolist())

        # planarity: fit a plane via SVD, RMS of signed distances to it
        centroid = pts3.mean(axis=0)
        normal = np.linalg.svd(pts3 - centroid)[2][-1]
        resid = (pts3 - centroid) @ normal
        planarity_mm.append(float(np.sqrt(np.mean(resid ** 2)) * 1000.0))

    if used == 0:
        raise SystemExit("no pairs with the board detected in BOTH halves — "
                         "check --cols/--rows and that pairs exist in the folder")

    sp = np.array(spacings_mm)
    plan = np.array(planarity_mm)
    err = (sp.mean() - square_mm) / square_mm * 100.0
    print(f"\nverify: {used} pairs, {len(sp)} corner-neighbour distances")
    print(f"  input square size    : {square_mm:.2f} mm")
    print(f"  reconstructed square : {sp.mean():.2f} mm  (std {sp.std():.2f})")
    print(f"  metric scale error   : {err:+.1f} %")
    print(f"  planarity RMS        : mean {plan.mean():.2f} mm  (worst {plan.max():.2f} mm)")
    ok = abs(err) < 3.0 and plan.mean() < 1.0
    if ok:
        print("  => scale and triangulation geometry look good.")
    else:
        if abs(err) >= 3.0:
            print("  => scale off. NOTE: this error is INVARIANT to --square-mm (scaling the\n"
                  "     square scales both objp and the recovered baseline, so the ratio is\n"
                  "     unchanged) — re-measuring the board will NOT move it. A persistent\n"
                  "     offset means a geometry/convention fault: check that the calibration\n"
                  "     carries R1/R2 and that undistort_points applies them (P1/P2 from\n"
                  "     stereoRectify are in the RECTIFIED frame).")
            if not calib.has_rectification():
                print("     !! this calibration has NO R1/R2 — recalibrate to regenerate it.")
        if plan.mean() >= 1.0:
            print("  => planarity poor: triangulation geometry suspect (baseline/rectification)\n"
                  "     — recheck stereo RMS and recapture with more depth/pose variation.")


def find_corners(img_gray, pattern):
    """Return refined internal-corner coords, or None if not fully detected.

    Primary: findChessboardCornersSB (sector-based) — robust to blur, glare,
    and minimal border, which is exactly the failure mode of a screen target.
    Fallback: legacy findChessboardCorners + cornerSubPix.
    """
    sb_flags = (cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE
                + cv2.CALIB_CB_ACCURACY)
    ok, corners = cv2.findChessboardCornersSB(img_gray, pattern, sb_flags)
    if ok:
        return corners  # SB returns sub-pixel corners already
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(img_gray, pattern, flags)
    if not ok:
        return None
    return cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), CRITERIA)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("data/calibration"),
                    help="folder with camera-1-NN.jpg / camera-2-NN.jpg pairs")
    ap.add_argument("--cols", type=int, required=True,
                    help="INTERNAL corners across (squares_wide - 1)")
    ap.add_argument("--rows", type=int, required=True,
                    help="INTERNAL corners down (squares_tall - 1)")
    ap.add_argument("--square-mm", type=float, required=True,
                    help="physical square size in mm (measure it on the tablet)")
    ap.add_argument("--out", type=Path, default=Path("assets/hardware/calibration"),
                    help="calibration output directory")
    ap.add_argument("--verify", action="store_true",
                    help="don't calibrate; triangulate board corners with the saved "
                         "calibration and report metric scale + planarity")
    args = ap.parse_args()

    pattern = (args.cols, args.rows)
    square_m = args.square_mm / 1000.0

    if args.verify:
        verify_calibration(args.dir, pattern, args.square_mm, args.out)
        return

    # canonical object points for one board, scaled to metres
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * square_m

    left_imgs = sorted(args.dir.glob("camera-1-*.jpg"))
    if not left_imgs:
        raise SystemExit(f"no camera-1-*.jpg found in {args.dir}")

    corner_dir = args.dir / "corners"
    corner_dir.mkdir(exist_ok=True)

    objpoints, imgpoints_l, imgpoints_r = [], [], []
    img_size = None
    used, skipped = 0, []
    for lp in left_imgs:
        rp = args.dir / lp.name.replace("camera-1-", "camera-2-")
        if not rp.exists():
            skipped.append((lp.name, "no right pair"));  continue
        il, ir = cv2.imread(str(lp)), cv2.imread(str(rp))
        gl, gr = cv2.cvtColor(il, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gl.shape[::-1]  # (w, h)
        cl, cr = find_corners(gl, pattern), find_corners(gr, pattern)
        if cl is None or cr is None:
            miss = "L" if cl is None else ""
            miss += "R" if cr is None else ""
            skipped.append((lp.name, f"corners not found ({miss})"));  continue
        objpoints.append(objp)
        imgpoints_l.append(cl)
        imgpoints_r.append(cr)
        used += 1
        # save an overlay for visual QC
        vis_l = cv2.drawChessboardCorners(il.copy(), pattern, cl, True)
        vis_r = cv2.drawChessboardCorners(ir.copy(), pattern, cr, True)
        cv2.imwrite(str(corner_dir / f"corners-{lp.stem}.jpg"),
                    cv2.hconcat([vis_l, vis_r]))

    print(f"\npairs used: {used} / {len(left_imgs)}")
    for name, why in skipped:
        print(f"  skipped {name}: {why}")
    print(f"corner overlays -> {corner_dir}/  (inspect, delete bad source pairs, re-run)")
    if used < 8:
        raise SystemExit(f"\nonly {used} good pairs — need ~15+ for a stable "
                         "calibration. Capture more, varying distance/position.")

    # per-camera intrinsics
    rms_l, K1, D1, *_ = cv2.calibrateCamera(objpoints, imgpoints_l, img_size, None, None)
    rms_r, K2, D2, *_ = cv2.calibrateCamera(objpoints, imgpoints_r, img_size, None, None)
    print(f"\nintrinsic RMS reproj error:  left={rms_l:.3f}px  right={rms_r:.3f}px")

    # stereo extrinsics (fix intrinsics we just solved)
    rms_s, K1, D1, K2, D2, R, T, *_ = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r, K1, D1, K2, D2, img_size,
        flags=cv2.CALIB_FIX_INTRINSIC, criteria=CRITERIA)
    print(f"stereo RMS reproj error:     {rms_s:.3f}px  (want < ~1.0)")

    # rectified projection matrices P1, P2 (3x4) — what triangulate_points uses
    R1, R2, P1, P2, Q, *_ = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T,
                                              flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    args.out.mkdir(parents=True, exist_ok=True)
    # R1/R2 are REQUIRED, not optional extras: P1/P2 are defined in the RECTIFIED
    # frame, so points must be undistorted with R=R1/R2 to land in that frame before
    # triangulation. Omitting them leaves points in the original camera frame and
    # produces a systematic scale error (~11% on this rig) that is invariant to
    # --square-mm, so it cannot be tuned away by re-measuring the board.
    stereo = {"camera-1-camera-2": {
        "cameraMatrix1": K1, "distCoeffs1": D1, "P1": P1, "R1": R1,
        "cameraMatrix2": K2, "distCoeffs2": D2, "P2": P2, "R2": R2,
        "R": R, "T": T, "Q": Q}}
    with open(args.out / "stereo_params.pickle", "wb") as fh:
        pickle.dump(stereo, fh)
    for cam, K, D in (("camera-1", K1, D1), ("camera-2", K2, D2)):
        with open(args.out / f"{cam}_intrinsic_params.pickle", "wb") as fh:
            pickle.dump({"cameraMatrix": K, "distCoeffs": D}, fh)

    print(f"\nwrote {args.out}/stereo_params.pickle (+ intrinsics).")
    print("Next: python -m shoggoth_mini debug-perception  and check the 3D points.")
    if rms_s > 1.0:
        print("!! stereo RMS > 1.0 — recalibrate: more pairs, less glare/moire, "
              "board fully visible in BOTH halves, correct --cols/--rows/--square-mm.")


if __name__ == "__main__":
    main()
