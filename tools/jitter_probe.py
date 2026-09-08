#!/usr/bin/env python
"""Apportion landmark jitter between the model, the ROI crop, and the sensor.

    python tools/jitter_probe.py                 # camera, hold still ~10 s
    python tools/jitter_probe.py --source v.mp4

Jitter is visible in the viewer, but "the landmarks wobble" does not say which
of three very different things is wobbling, and each has a different fix:

  A  THE MODEL, given identical pixels. FaceLandmarker is a deterministic
     regressor, so this should be ~0. If it is not, nothing downstream can be
     smoothed into correctness and the only recourse is temporal filtering.

  B  THE ROI CROP. Roi.follow() recomputes x0/y0/side from the landmark bounding
     box EVERY frame and rounds to int, so a still head is fed a crop that
     shifts by a pixel or two and rescales slightly each time. The detector then
     sees a differently-framed, differently-scaled image and answers differently.
     This is ours, not MediaPipe's, and it is fixable by smoothing the box.

  C  THE SENSOR AND THE SCENE. Photon noise, auto-exposure, real micro-motion.
     Irreducible; only temporal filtering helps.

The test freezes ONE frame and re-runs detection on it, which is the only way to
separate these: identical pixels isolate A, deliberately shifted crops measure
B, and live frames give A+B+C together. What is left after subtracting is C.

Reports in DEGREES of yaw/pitch, because that is the unit the attention test
and the gesture detector actually consume -- a landmark wobble of half a pixel
matters only insofar as it moves the number a threshold is compared against.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from shoggoth_mini.perception.camera import read_oriented, set_camera_orientation  # noqa: E402
from shoggoth_mini.perception.face import (Roi, SEARCH_CROP,  # noqa: E402
                                           detect_face)
from shoggoth_mini.perception.stereo import split_stereo_frame  # noqa: E402
from shoggoth_mini.configs import get_perception_config  # noqa: E402
from shoggoth_mini.affect import ATTEND_PITCH_DEG, ATTEND_YAW_DEG  # noqa: E402

PKG = HERE.parent / "shoggoth_mini"


def stats(name: str, yaw: list, pitch: list, px: list) -> dict:
    """Spread of a set of detections that SHOULD have been identical."""
    y, p = np.array(yaw), np.array(pitch)
    d = {"n": len(y), "yaw_sd": float(y.std()), "pitch_sd": float(p.std()),
         "yaw_pp": float(y.max() - y.min()) if len(y) else 0.0,
         "pitch_pp": float(p.max() - p.min()) if len(p) else 0.0}
    if px:
        stack = np.stack(px)                      # (n, 478, 2)
        # per-landmark scatter, then the typical one across the mesh
        d["px_sd"] = float(np.median(stack.std(axis=0).mean(axis=1)))
    else:
        d["px_sd"] = float("nan")
    print(f"  {name:34} n={d['n']:3}  yaw sd {d['yaw_sd']:6.3f} deg "
          f"(p-p {d['yaw_pp']:5.2f})  pitch sd {d['pitch_sd']:6.3f}  "
          f"landmark sd {d['px_sd']:5.2f} px")
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0")
    ap.add_argument("--repeats", type=int, default=40,
                    help="detections per frozen-frame test")
    ap.add_argument("--live-seconds", type=float, default=10.0)
    ap.add_argument("--crop", type=int, default=SEARCH_CROP)
    ap.add_argument("--save-frame", type=Path, default=None,
                    help="write the frozen frame, so the test can be repeated")
    args = ap.parse_args()

    yaml = PKG / "configs" / "default_perception.yaml"
    cfg = get_perception_config(str(yaml) if yaml.exists() else None)
    set_camera_orientation(cfg.camera_rotate_180)

    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.source!r}")
    if isinstance(src, int):
        w, h = cfg.stereo_resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    print("\nHOLD STILL. Looking for a face...")
    frozen = roi0 = None
    for _ in range(200):
        ok, frame = read_oriented(cap)
        if not ok:
            break
        left, _ = split_stereo_frame(frame)
        r = Roi(left.shape[1], left.shape[0], args.crop)
        if detect_face(left, r) is not None:
            frozen, roi0 = left.copy(), r
            break
    if frozen is None:
        cap.release()
        raise SystemExit("no face found; nothing to measure")
    print(f"  got one. frame {frozen.shape[1]}x{frozen.shape[0]}, "
          f"roi {roi0.side}px at ({roi0.x0},{roi0.y0})\n")
    if args.save_frame:
        cv2.imwrite(str(args.save_frame), frozen)

    # ---- A: identical pixels, identical crop -------------------------------
    print("A  MODEL, identical pixels and identical crop")
    ys, ps, pxs = [], [], []
    for _ in range(args.repeats):
        r = Roi(frozen.shape[1], frozen.shape[0], args.crop)
        r.x0, r.y0, r.side, r.locked = roi0.x0, roi0.y0, roi0.side, True
        o = detect_face(frozen, r)
        if o:
            ys.append(o.yaw); ps.append(o.pitch); pxs.append(o.px)
    a = stats("same frame, same crop", ys, ps, pxs)

    # ---- B: identical pixels, crop jogged the way follow() jogs it ---------
    print("\nB  ROI CROP, same pixels but the box moved +/-2 px and +/-1% scale")
    ys, ps, pxs = [], [], []
    rng = np.random.default_rng(0)
    for _ in range(args.repeats):
        r = Roi(frozen.shape[1], frozen.shape[0], args.crop)
        side = int(roi0.side * (1.0 + rng.uniform(-0.01, 0.01)))
        r.side = int(np.clip(side, 1, min(r.fw, r.fh)))
        r.x0 = int(np.clip(roi0.x0 + rng.integers(-2, 3), 0, r.fw - r.side))
        r.y0 = int(np.clip(roi0.y0 + rng.integers(-2, 3), 0, r.fh - r.side))
        r.locked = True
        o = detect_face(frozen, r)
        if o:
            ys.append(o.yaw); ps.append(o.pitch); pxs.append(o.px)
    b = stats("same frame, crop jogged", ys, ps, pxs)

    # ---- C: live, everything moving ---------------------------------------
    print(f"\nC  LIVE, {args.live_seconds:.0f}s of real frames (still keeping still)")
    ys, ps, pxs = [], [], []
    roi = Roi(frozen.shape[1], frozen.shape[0], args.crop)
    t0 = time.time()
    while time.time() - t0 < args.live_seconds:
        ok, frame = read_oriented(cap)
        if not ok:
            break
        left, _ = split_stereo_frame(frame)
        o = detect_face(left, roi)
        if o:
            ys.append(o.yaw); ps.append(o.pitch); pxs.append(o.px)
    cap.release()
    c = stats("live frames", ys, ps, pxs)

    # ---- what it means ----------------------------------------------------
    print("\n" + "=" * 72)
    if a["yaw_sd"] < 1e-9 and a["pitch_sd"] < 1e-9:
        print("A = 0: the model is DETERMINISTIC. Every wobble is the input "
              "changing,\n       so smoothing the input is a real fix.")
    else:
        print(f"A > 0: the model itself varies on identical pixels "
              f"({a['yaw_sd']:.3f} deg).\n       Only temporal filtering helps.")
    if b["yaw_sd"] > max(a["yaw_sd"], 1e-9) * 2:
        print(f"B >> A: a 2 px crop shift moves yaw by {b['yaw_sd']:.3f} deg sd. "
              f"Roi.follow()\n        rounds to int every frame, so this fires "
              f"constantly. OURS to fix.")
    else:
        print("B ~ A: the crop is not a significant contributor.")
    live = c["yaw_sd"]
    print(f"\nLive yaw sd {live:.3f} deg vs attention threshold "
          f"{ATTEND_YAW_DEG:.0f} deg (pitch {ATTEND_PITCH_DEG:.0f}).")
    print(f"A head parked within {2*live:.2f} deg of a threshold will chatter, "
          f"because\nattending() is an instantaneous comparison with no "
          f"hysteresis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
