"""Generate a known-good checkerboard target to display on the tablet.

Produces an ASYMMETRIC board (interior corners of different parity -> unique
origin, so corner ordering is stable across frames) with a fat white quiet
zone (the detector needs a light border around the outer squares).

Default: 10x7 squares  ->  9x6 INTERNAL corners  ->  calibrate with --cols 9 --rows 6

  python tools/make_checkerboard.py --out data/calibration/target_9x6.png

Display it FULL-SCREEN but DO NOT stretch (preserve aspect — squares must stay
square). Then capture with `record stereo-calibration --manual --cols 9 --rows 6`.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--squares-x", type=int, default=10, help="squares across (even)")
    ap.add_argument("--squares-y", type=int, default=7, help="squares down (odd)")
    ap.add_argument("--square-px", type=int, default=140)
    ap.add_argument("--border-px", type=int, default=180, help="white quiet zone")
    ap.add_argument("--out", type=Path, default=Path("data/calibration/target_9x6.png"))
    args = ap.parse_args()

    sx, sy, s, b = args.squares_x, args.squares_y, args.square_px, args.border_px
    if (sx % 2) == (sy % 2):
        print("WARNING: squares-x and squares-y have the same parity -> the interior\n"
              "corner grid is rotationally ambiguous. Use one even + one odd.")

    canvas = np.full((sy * s + 2 * b, sx * s + 2 * b), 255, np.uint8)
    for j in range(sy):
        for i in range(sx):
            if (i + j) % 2 == 0:
                y0, x0 = b + j * s, b + i * s
                canvas[y0:y0 + s, x0:x0 + s] = 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), canvas)
    print(f"wrote {args.out}  ({canvas.shape[1]}x{canvas.shape[0]} px)")
    print(f"board: {sx}x{sy} squares -> {sx-1}x{sy-1} INTERNAL corners")
    print(f"=> calibrate/capture with --cols {sx-1} --rows {sy-1}")


if __name__ == "__main__":
    main()
