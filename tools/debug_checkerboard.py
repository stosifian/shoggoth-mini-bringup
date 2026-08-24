"""Diagnose why checkerboard detection is failing on a calibration image.

Sweeps a range of INTERNAL-corner dimensions and reports which (if any) the
OpenCV detectors accept — this reveals whether the problem is a wrong corner
count vs. image quality. Saves overlays + preprocessing views for inspection.

Run from repo root, venv active:
  python tools/debug_checkerboard.py data/calibration/camera-1-01.jpg --cols 9 --rows 6

Outputs go to <image_dir>/debug/:
  gray.jpg, thresh.jpg      — what the detector sees / binarized contrast
  detect_CxR.jpg            — one per pattern size that successfully detected
"""
import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np


def try_sb(gray, pattern):
    ok, c = cv2.findChessboardCornersSB(
        gray, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE)
    return (c if ok else None)


def try_legacy(gray, pattern):
    ok, c = cv2.findChessboardCorners(
        gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    return (c if ok else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("--cols", type=int, default=9, help="your EXPECTED corners across")
    ap.add_argument("--rows", type=int, default=6, help="your EXPECTED corners down")
    ap.add_argument("--min", type=int, default=4, help="min corners per axis to sweep")
    ap.add_argument("--max", type=int, default=12, help="max corners per axis to sweep")
    args = ap.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"could not read {args.image}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"image {args.image.name}: {w}x{h}, "
          f"intensity min/mean/max = {gray.min()}/{gray.mean():.0f}/{gray.max()}")

    out = args.image.parent / "debug"
    out.mkdir(exist_ok=True)
    cv2.imwrite(str(out / "gray.jpg"), gray)
    # adaptive threshold: shows whether squares separate cleanly (detector relies on this)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 51, 5)
    cv2.imwrite(str(out / "thresh.jpg"), thr)

    # 1) explicit test of the size you expect, both detectors
    print(f"\nexpected size {args.cols}x{args.rows}:")
    for name, fn in (("findChessboardCornersSB", try_sb),
                     ("findChessboardCorners  ", try_legacy)):
        c = fn(gray, (args.cols, args.rows))
        print(f"  {name}: {'DETECTED' if c is not None else 'no'}")
        if c is not None:
            vis = cv2.drawChessboardCorners(img.copy(), (args.cols, args.rows), c, True)
            cv2.imwrite(str(out / f"detect_{args.cols}x{args.rows}_{name.strip()}.jpg"), vis)

    # 2) sweep to discover the true count if the expected one fails
    print(f"\nsweeping SB over {args.min}..{args.max} corners per axis "
          "(this finds the ACTUAL grid if the expected count is wrong)...")
    hits = []
    for c, r in itertools.product(range(args.min, args.max + 1),
                                  range(args.min, args.max + 1)):
        corners = try_sb(gray, (c, r))
        if corners is not None:
            hits.append((c, r))
            vis = cv2.drawChessboardCorners(img.copy(), (c, r), corners, True)
            cv2.imwrite(str(out / f"detect_{c}x{r}.jpg"), vis)
            print(f"  DETECTED {c}x{r}  -> saved detect_{c}x{r}.jpg")

    print("\n--- summary ---")
    if not hits:
        print("No pattern size detected. Likely causes: board clipped/curved at the\n"
              "edges (wide-FOV barrel distortion breaking the outer corner topology),\n"
              "low local contrast, or reflections. Inspect debug/thresh.jpg — the\n"
              "squares must separate into clean alternating blocks to the very edge.")
    else:
        print(f"Detected sizes: {hits}")
        if (args.cols, args.rows) not in hits and (args.rows, args.cols) not in hits:
            print(f"** Your --cols {args.cols} --rows {args.rows} is likely WRONG. "
                  f"Use one of the detected sizes above for --cols/--rows. **")
        print(f"Overlays in {out}/ — open detect_*.jpg to confirm corners land on\n"
              "real intersections and cover the whole board.")


if __name__ == "__main__":
    main()
