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
        d["px_max"] = float(stack.std(axis=0).max())
        # A zero is only believable if the samples are genuinely separate
        # objects. Appending one aliased array 40 times would also report 0.
        d["distinct"] = len({id(a) for a in px})
    else:
        d["px_sd"] = d["px_max"] = float("nan")
        d["distinct"] = 0
    # Full precision, not %.3f. A rounded 0.000 hides 4e-4, and the whole
    # question about test A is whether the zero is exact or merely small.
    print(f"  {name:34} n={d['n']:3}  yaw sd {d['yaw_sd']:.6g} deg "
          f"(p-p {d['yaw_pp']:.4g})  pitch sd {d['pitch_sd']:.6g}  "
          f"landmark sd {d['px_sd']:.4g} px (worst {d['px_max']:.4g}, "
          f"{d['distinct']}/{d['n']} distinct arrays)")
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

    # ---- A': positive control ---------------------------------------------
    # A zero from test A is worthless on its own: a harness that always returns
    # the same number would report exactly this. So perturb the pixels by a
    # single least-significant bit -- far below anything visible, and far below
    # real sensor noise -- and require the output to MOVE. If it does not, the
    # zero above is measuring the harness rather than the model.
    print("\nA' CONTROL, same crop but +/-1 LSB of pixel noise (must be > 0)")
    ys, ps, pxs = [], [], []
    rng = np.random.default_rng(1)
    for _ in range(args.repeats):
        noisy = np.clip(frozen.astype(np.int16)
                        + rng.integers(-1, 2, frozen.shape, dtype=np.int16),
                        0, 255).astype(np.uint8)
        r = Roi(frozen.shape[1], frozen.shape[0], args.crop)
        r.x0, r.y0, r.side, r.locked = roi0.x0, roi0.y0, roi0.side, True
        o = detect_face(noisy, r)
        if o:
            ys.append(o.yaw); ps.append(o.pitch); pxs.append(o.px)
    ctrl = stats("1 LSB of noise", ys, ps, pxs)
    if ctrl["px_sd"] <= 0.0 and ctrl["yaw_sd"] <= 0.0:
        print("  !! CONTROL FAILED: one bit of pixel noise changed nothing.")
        print("     Test A's zero is NOT evidence about the model -- something")
        print("     in this harness is returning a constant. Do not trust A.")
    else:
        print(f"  control moved (landmark sd {ctrl['px_sd']:.4g} px), so the "
              f"harness CAN see a\n  difference, and test A's zero is real.")

    # ---- C: live, everything moving ---------------------------------------
    # Runs BEFORE B, because B needs to know how far the box really moves.
    # detect_face calls roi.follow() in place, so reading the box back after
    # each detection is the actual production trajectory -- no instrumentation
    # inside face.py, and nothing that could alter what it measures.
    print(f"\nC  LIVE, {args.live_seconds:.0f}s of real frames (still keeping still)")
    ys, ps, pxs, boxes = [], [], [], []
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
            boxes.append((roi.x0, roi.y0, roi.side))
    cap.release()
    c = stats("live frames", ys, ps, pxs)

    # ---- how far does the box ACTUALLY move? -------------------------------
    bx = np.array(boxes, float)
    if len(bx) < 3:
        raise SystemExit("too few live detections to measure ROI motion")
    d = np.diff(bx, axis=0)                       # per-frame dx, dy, dside
    dscale = d[:, 2] / bx[:-1, 2]
    print(f"\n   measured ROI motion over {len(bx)} frames:")
    print(f"     dx    median {np.median(np.abs(d[:,0])):5.1f} px  "
          f"p95 {np.percentile(np.abs(d[:,0]),95):5.1f}  max {np.abs(d[:,0]).max():5.0f}")
    print(f"     dy    median {np.median(np.abs(d[:,1])):5.1f} px  "
          f"p95 {np.percentile(np.abs(d[:,1]),95):5.1f}  max {np.abs(d[:,1]).max():5.0f}")
    print(f"     dside median {np.median(np.abs(dscale))*100:5.2f} %   "
          f"p95 {np.percentile(np.abs(dscale),95)*100:5.2f}   "
          f"max {np.abs(dscale).max()*100:5.1f}")

    # ---- B: identical pixels, crop jogged by the MEASURED distribution -----
    # The first version of this test assumed +/-2 px and +/-1 %. That was a
    # guess, and two runs disagreed by 9x on how much of the live pitch
    # variance it explained -- because the effect scales with crop size, which
    # changes every sitting. Resampling the real per-frame steps removes the
    # assumption: whatever the box did during THIS live pass is what gets
    # replayed against a frozen frame.
    print("\nB  ROI CROP, same pixels, box jogged by the MEASURED steps above")
    ys, ps, pxs = [], [], []
    rng = np.random.default_rng(0)
    for _ in range(args.repeats):
        dx, dy, ds = d[rng.integers(0, len(d))]
        r = Roi(frozen.shape[1], frozen.shape[0], args.crop)
        r.side = int(np.clip(roi0.side + ds, 16, min(r.fw, r.fh)))
        r.x0 = int(np.clip(roi0.x0 + dx, 0, r.fw - r.side))
        r.y0 = int(np.clip(roi0.y0 + dy, 0, r.fh - r.side))
        r.locked = True
        o = detect_face(frozen, r)
        if o:
            ys.append(o.yaw); ps.append(o.pitch); pxs.append(o.px)
    b = stats("frozen frame, real box steps", ys, ps, pxs)

    # ---- what it means ----------------------------------------------------
    print("\n" + "=" * 72)
    if a["yaw_sd"] < 1e-9 and a["pitch_sd"] < 1e-9:
        if ctrl["px_sd"] > 0.0 or ctrl["yaw_sd"] > 0.0:
            print("A = 0 and the control moved: the model is DETERMINISTIC. "
                  "Every wobble is\n       the input changing, so smoothing "
                  "the input is a real fix.")
        else:
            print("A = 0 but the CONTROL ALSO = 0. This says nothing about the "
                  "model;\n       the harness is broken. Fix it before reading "
                  "B or C.")
    else:
        print(f"A > 0: the model itself varies on identical pixels "
              f"({a['yaw_sd']:.3f} deg).\n       Only temporal filtering helps.")
    # Variance is what adds, so apportion there and report as a share of live.
    # A share near or above 100% does NOT mean the crop explains everything --
    # it means the frozen-frame replay is not a clean subset of the live
    # condition, and the split should not be quoted.
    print("\n  share of LIVE variance, by axis:")
    for ax in ("yaw", "pitch"):
        cv = c[f"{ax}_sd"] ** 2
        if cv <= 0:
            continue
        crop = b[f"{ax}_sd"] ** 2 / cv
        lsb = ctrl[f"{ax}_sd"] ** 2 / cv
        rest = 1.0 - crop
        flag = "  <-- see note" if crop > 0.85 else ""
        print(f"    {ax:5}  crop {crop*100:5.1f}%   sensor+motion "
              f"{max(rest,0)*100:5.1f}%   (1 LSB alone would be "
              f"{lsb*100:4.1f}%){flag}")
    if max(b["yaw_sd"] ** 2 / max(c["yaw_sd"] ** 2, 1e-12),
           b["pitch_sd"] ** 2 / max(c["pitch_sd"] ** 2, 1e-12)) > 0.85:
        print("\n  NOTE: a crop share near 100% means B is not cleanly nested "
              "inside C --\n  the frozen frame differs from the live scene in "
              "lighting or pose. Repeat\n  the run before quoting the split; "
              "these shares have varied 9x between\n  sittings.")
    live = c["yaw_sd"]
    print(f"\nLive yaw sd {live:.3f} deg vs attention threshold "
          f"{ATTEND_YAW_DEG:.0f} deg (pitch {ATTEND_PITCH_DEG:.0f}).")
    print(f"A head parked within {2*live:.2f} deg of a threshold will chatter, "
          f"because\nattending() is an instantaneous comparison with no "
          f"hysteresis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
