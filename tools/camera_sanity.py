"""Sanity-check the stereo USB camera before any robot bring-up.

Verifies the substitute ELP camera against the procurement criteria:
  * enumerates as ONE UVC device at the given index,
  * delivers the requested side-by-side resolution (default 3840x1080),
  * the two halves are a genuine STEREO pair (different viewpoints, not a
    duplicated/mono feed), and are hardware-SYNCHRONIZED (both move together).

Nothing here touches the robot — camera only. Run from repo root, venv active:

  # FIRST: list cameras and find the USB stereo one (index 0 is usually the
  # Mac's built-in FaceTime cam; the stereo cam is a higher index with a
  # very wide aspect ratio when asked for a wide mode).
  python tools/camera_sanity.py --list

  # then live view on the index the list flagged as stereo, e.g. 1:
  # wave your hand across the lens, both halves must move as one
  python tools/camera_sanity.py --camera 1

  # headless (e.g. over SSH): print stats for 120 frames, no window
  python tools/camera_sanity.py --no-display --frames 120

Keys in live view:  s = save snapshot pair   q / Esc = quit
"""
import argparse
import time
from pathlib import Path

import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.perception.camera import read_oriented  # noqa: E402


def fourcc_str(cap) -> str:
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4)).strip() or "?"


def split_sbs(frame):
    """Split a side-by-side frame into (left, right) halves."""
    half = frame.shape[1] // 2
    return frame[:, :half], frame[:, half:]


def list_cameras(max_index, width, height):
    """Probe indices 0..max_index and print what each reports.

    The USB stereo cam is identified by a very wide aspect ratio (~32:9) when
    asked for a wide SBS mode; the Mac's built-in FaceTime cam clamps to 16:9.
    """
    print(f"probing camera indices 0..{max_index} "
          f"(requesting {width}x{height})...\n")
    print("index  actual-res    aspect  fourcc  guess")
    found_stereo = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = read_oriented(cap)
        if not ok or frame is None:
            print(f"{i:5d}  (opens but no frame)")
            cap.release()
            continue
        h, w = frame.shape[:2]
        aspect = w / h if h else 0.0
        fcc = fourcc_str(cap)
        stereo = aspect > 2.5
        if stereo:
            found_stereo.append(i)
        print(f"{i:5d}  {w:4d}x{h:<5d}  {aspect:5.2f}  {fcc:>5}   "
              f"{'<-- likely STEREO (SBS)' if stereo else 'built-in / mono'}")
        cap.release()
    print()
    if found_stereo:
        print(f"-> use:  python tools/camera_sanity.py --camera {found_stereo[0]}")
    else:
        print("-> no wide/SBS camera found. Is the USB cam plugged in and granted\n"
              "   camera permission (System Settings > Privacy & Security > Camera)?\n"
              "   Try raising --max-index, or unplug the built-in test by index.")


def open_camera(index, width, height):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: could not open camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"requested : {width}x{height}")
    print(f"actual    : {aw}x{ah}   fourcc={fourcc_str(cap)}   "
          f"driver-fps={cap.get(cv2.CAP_PROP_FPS):.1f}")
    if (aw, ah) != (width, height):
        print("  ! actual != requested — the camera fell back to a supported mode.")
    if aw % 2:
        print("  ! odd width — cannot split cleanly into L/R halves.")
    per_eye = (aw // 2, ah)
    print(f"per-eye   : {per_eye[0]}x{per_eye[1]} after SBS split")
    return cap


def stereo_report(left, right):
    """One-line heuristics: are the halves a real, distinct stereo pair?"""
    l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    lr_diff = float(np.mean(cv2.absdiff(l, r)))
    verdict = "identical(!) — mono/duplicated feed?" if lr_diff < 1.0 else "distinct viewpoints — OK"
    return lr_diff, verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="probe indices and identify the USB stereo camera, then exit")
    ap.add_argument("--max-index", type=int, default=6,
                    help="highest index to probe in --list mode")
    ap.add_argument("--camera", type=int, default=0, help="camera device index")
    ap.add_argument("--width", type=int, default=3840, help="requested SBS frame width")
    ap.add_argument("--height", type=int, default=1080, help="requested frame height")
    ap.add_argument("--no-display", action="store_true", help="headless: print stats only")
    ap.add_argument("--frames", type=int, default=0,
                    help="stop after N frames (0 = run until quit)")
    ap.add_argument("--out", default="camera_sanity", help="snapshot filename prefix")
    ap.add_argument("--view-width", type=int, default=1280,
                    help="downscaled width of the live preview window")
    args = ap.parse_args()

    if args.list:
        list_cameras(args.max_index, args.width, args.height)
        return

    cap = open_camera(args.camera, args.width, args.height)

    # measured FPS over a rolling window + motion between consecutive frames
    prev_gray = None
    t_prev = time.time()
    fps = 0.0
    n = 0
    print("\nframe  meas-fps  L|R-diff  frame-to-frame-motion  stereo-verdict")
    try:
        while True:
            ok, frame = read_oriented(cap)
            if not ok:
                print("ERROR: frame grab failed"); break
            n += 1

            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            left, right = split_sbs(frame)
            lr_diff, verdict = stereo_report(left, right)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion = 0.0 if prev_gray is None else float(np.mean(cv2.absdiff(gray, prev_gray)))
            prev_gray = gray

            if n % 10 == 0 or args.no_display:
                print(f"{n:5d}  {fps:7.1f}  {lr_diff:7.1f}  {motion:19.2f}  {verdict}")

            if not args.no_display:
                combo = cv2.hconcat([left, right])
                scale = args.view_width / combo.shape[1]
                view = cv2.resize(combo, (args.view_width, int(combo.shape[0] * scale)))
                cv2.line(view, (view.shape[1] // 2, 0),
                         (view.shape[1] // 2, view.shape[0]), (0, 255, 255), 1)
                cv2.putText(view, f"{fps:.1f} fps  L|R-diff={lr_diff:.0f}  {verdict}",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.imshow("stereo sanity  (left | right)", view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    Path(f"{args.out}_full.png").write_bytes(cv2.imencode(".png", frame)[1])
                    cv2.imwrite(f"{args.out}_left.png", left)
                    cv2.imwrite(f"{args.out}_right.png", right)
                    print(f"  saved {args.out}_full/left/right.png")

            if args.frames and n >= args.frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\ndone: {n} frames, ~{fps:.1f} fps steady-state.")
    print("PASS criteria: actual==requested resolution, steady fps near driver-fps,")
    print("L|R-diff comfortably > 1 (distinct views), and both halves move together")
    print("with no visible lag when you wave a hand across the lens (hardware sync).")


if __name__ == "__main__":
    main()
