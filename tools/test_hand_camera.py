"""Live camera test for the ported MediaPipe hand tracking.

Exercises ONLY perception/hand_tracking.py (no YOLO/triangulation), so it's a
clean check that the Tasks-API port works on your actual camera. Opens the
stereo camera, uses the left half of the side-by-side frame, overlays the
detected hand skeleton + index-fingertip, and shows measured FPS.

Run from repo root, venv active:
  python tools/test_hand_camera.py --camera 0

Wave/move your hand in view: the green skeleton + red dots should track it.
Keys:  q / Esc = quit
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
from shoggoth_mini.perception.camera import read_oriented

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.perception.hand_tracking import (  # noqa: E402
    get_mediapipe_hand_data,
    draw_hand_landmarks,
    close_mediapipe_hands,
    INDEX_FINGER_TIP,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="camera device index")
    ap.add_argument("--width", type=int, default=3840, help="requested SBS width")
    ap.add_argument("--height", type=int, default=1080, help="requested height")
    ap.add_argument("--view-width", type=int, default=960, help="preview width")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    fps, t_prev = 0.0, time.time()
    try:
        while True:
            ok, frame = read_oriented(cap)
            if not ok:
                print("frame grab failed"); break

            # use the left half of a side-by-side stereo frame
            h, w = frame.shape[:2]
            left = frame[:, : w // 2] if w / h > 2.5 else frame

            tip, results = get_mediapipe_hand_data(left)
            vis = draw_hand_landmarks(left, results)
            if tip is not None:
                cv2.circle(vis, (int(tip[0]), int(tip[1])), 8, (0, 255, 255), 2)

            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt
            status = f"{fps:4.1f} fps   hand: {'YES' if tip is not None else 'no'}"
            cv2.putText(vis, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2, cv2.LINE_AA)

            scale = args.view_width / vis.shape[1]
            view = cv2.resize(vis, (args.view_width, int(vis.shape[0] * scale)))
            cv2.imshow("hand tracking (left eye)", view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        close_mediapipe_hands()


if __name__ == "__main__":
    main()
