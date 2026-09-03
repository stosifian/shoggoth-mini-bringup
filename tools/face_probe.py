"""Stereo face perception bench -- landmarks, 3-D head position, expression channels.

DESIGN/RESEARCH (see NEXT_PHASE_EXPLORATION.md). Read-only w.r.t. the POC: this
imports the perception stack but touches no motor and writes nothing the robot
reads. Nothing here can move the tentacle.

What it is for: the ELEGNT framework (arXiv 2501.12493) splits expressive motion
into ATTENTION (where the robot looks), EMOTION (how it moves), INTENTION and
ATTITUDE. The first two need perception this project does not have yet. This
answers whether that perception is good enough to drive anything, BEFORE any of
it is allowed near a servo.

    python exploration/face_probe.py                    # live, both eyes + panel
    python exploration/face_probe.py --headless --csv out/face.csv
    python exploration/face_probe.py --source clip.mp4  # no camera contention
    python exploration/face_probe.py --flip-yaw         # see 'head pose' below

MediaPipe is one library call. The part being prototyped is everything after it:

  * TRIANGULATION. Landmarks are normalised per-eye and scale-free. Running the
    two views through the project's own stereo calibration is what turns a face
    into a referent in METRES -- which is what "someone leaned in" needs to be a
    measurable event rather than a guess.
  * DERIVED CHANNELS. arousal/valence are weighted blendshape sums. The weights
    below are a STARTING GUESS, not a validated affect model -- see the note on
    BLENDSHAPE_WEIGHTS.
  * TIME. Single-frame values are close to useless for the appraisal layer.
    Dwell, approach speed and expression CHANGE are the signals with content.

Sanity check worth watching: the panel reports measured interpupillary distance.
A human adult IPD is ~63 mm (range roughly 55-72). If the number on screen is
far outside that, the stereo scale is wrong AT FACE RANGE and every 3-D value
here is wrong with it -- the calibration was fitted around the tentacle, much
closer to the cameras than a person's head sits.

HEAD POSE IS UNVERIFIED. The yaw/pitch signs depend on the sign convention of
MediaPipe's transformation matrix and on how the rig is mounted, and could not
be checked without a face in front of the cameras. Turn your head right and
confirm yaw goes positive; nod down and confirm pitch goes positive. If either
is backwards, pass --flip-yaw / --flip-pitch and record it in the config.
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

from shoggoth_mini.perception.camera import read_oriented          # noqa: E402
from shoggoth_mini.perception.stereo import (                      # noqa: E402
    load_stereo_calibration,
    split_stereo_frame,
    triangulate_points,
)
from shoggoth_mini.configs import get_perception_config            # noqa: E402
from shoggoth_mini.perception.face import (FaceObs, Roi,  # noqa: E402
                                           detect_face, IRIS_L, IRIS_R,
                                           MODEL_PATH, NOSE_TIP, ROI_MIN,
                                           ROI_MARGIN, SEARCH_CROP)
from shoggoth_mini.affect.channels import Channels  # noqa: E402
from shoggoth_mini.affect import (ATTEND_PITCH_DEG, ATTEND_YAW_DEG,  # noqa: E402
                                  BLENDSHAPE_WEIGHTS, STATE_BGR, AffectState,
                                  affect_from_blend, attending, head_angles)
from shoggoth_mini.affect.config import (BASELINE_MIN_S,  # noqa: E402
                                         BASELINE_WINDOW_S, DEADBAND,
                                         DEADBAND_HYST, STATE_QUADRANT,
                                         A_THRESH, V_THRESH)


# The project YAML must be named EXPLICITLY. get_perception_config() with no
# argument returns pydantic field defaults and silently ignores it -- the trap
# perception/camera.py documents. Getting this wrong here cost a session: the
# default units_to_meters is 0.05 against the YAML's 1.0, so every triangulated
# distance came out 20x small and a 63 mm IPD read as 3.1 mm.
DEFAULT_PERCEPTION_YAML = PKG / "shoggoth_mini" / "configs" / "default_perception.yaml"

# MediaPipe's face detector runs on a ~192 px square. A full 1920x760 eye is
# therefore scaled by 0.1, so a 150 mm face at 0.5 m (186 px at fx=620) reaches
# the detector as ~19 px -- right at BlazeFace's floor, which is why detection
# dies just past half a metre. Cropping to a square recovers the pixels: 760 px
# gives 0.25 scale and ~47 px on the same face.

# 478-landmark layout: 0..467 mesh, 468..472 left iris, 473..477 right iris.
# "Left" is MediaPipe's naming (the subject's left), not the image side.

# Face-range triangulation box. triangulate_points CLIPS to coordinate_limits
# and indexes it unconditionally, so None is not an option -- and the project's
# configured limits are the tentacle's workspace, a box a face never enters.
# Clipping to it would silently pin every head position to a corner.
FACE_LIMITS = {"X": {"clip_min": -3.0, "clip_max": 3.0},
               "Y": {"clip_min": -3.0, "clip_max": 3.0},
               "Z": {"clip_min": -3.0, "clip_max": 3.0}}

IPD_NOMINAL_M = 0.063



# =============================================================================
# detection
# =============================================================================


# =============================================================================
# derived channels -- the part with time in it
# =============================================================================


# =============================================================================
# overlays
# =============================================================================
MESH_C, IRIS_C, RAY_C = (90, 200, 255), (80, 255, 140), (60, 220, 255)


def draw_face(frame: np.ndarray, obs: Optional[FaceObs], eye: str,
              roi: Optional["Roi"] = None) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.putText(out, eye, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    if roi is not None:                       # show what the detector actually sees
        cv2.rectangle(out, (roi.x0, roi.y0), (roi.x0 + roi.side, roi.y0 + roi.side),
                      (90, 200, 120) if roi.locked else (110, 110, 110), 2)
        cv2.putText(out, f"roi {roi.side}px{' lock' if roi.locked else ''}",
                    (roi.x0 + 6, roi.y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (90, 200, 120) if roi.locked else (130, 130, 130), 1)
    if obs is None:
        cv2.putText(out, "no face", (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (70, 70, 220), 1)
        return out

    for x, y in obs.px[:468].astype(int):        # mesh
        cv2.circle(out, (x, y), 1, MESH_C, -1)
    for idx in (IRIS_L, IRIS_R):                 # iris centres
        if idx < len(obs.px):
            cv2.circle(out, tuple(obs.px[idx].astype(int)), 3, IRIS_C, -1)

    # head-pose ray, drawn from the nose in image space
    nose = obs.px[NOSE_TIP].astype(int)
    span = 0.28 * w
    tipx = int(nose[0] + span * np.sin(np.radians(obs.yaw)))
    tipy = int(nose[1] - span * np.sin(np.radians(obs.pitch)))
    cv2.arrowedLine(out, tuple(nose), (tipx, tipy), RAY_C, 2, tipLength=0.25)
    cv2.putText(out, f"yaw {obs.yaw:+5.1f}  pitch {obs.pitch:+5.1f}",
                (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RAY_C, 1)
    return out


WIN = "face probe"


def draw_stop_button(img: np.ndarray) -> tuple[int, int, int, int]:
    """Draw STOP top-right and return its (x, y, w, h) hit box.

    Drawn AFTER any display downscale, so the rectangle is in the same
    coordinate space the mouse callback reports clicks in. Draw it before
    scaling and the hit box lands somewhere else on a resized view.
    """
    h, w = img.shape[:2]
    bw, bh, pad = 96, 32, 12
    x0, y0 = w - bw - pad, pad
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (32, 32, 38), -1)
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (70, 70, 220), 2)
    cv2.putText(img, "STOP", (x0 + 17, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.60, (80, 80, 240), 2, cv2.LINE_AA)
    return x0, y0, bw, bh


def _bar(img, x, y, w, h, frac, color, signed=False, lw=1):
    cv2.rectangle(img, (x, y), (x + w, y + h), (55, 55, 55), lw)
    if signed:
        mid = x + w // 2
        end = int(mid + frac * w / 2)
        cv2.rectangle(img, (min(mid, end), y + lw), (max(mid, end), y + h - lw),
                      color, -1)
        cv2.line(img, (mid, y), (mid, y + h), (110, 110, 110), lw)
    else:
        cv2.rectangle(img, (x + lw, y + lw),
                      (x + lw + int(max(0.0, min(1.0, frac)) * (w - 2 * lw)),
                       y + h - lw),
                      color, -1)


def draw_panel(width: int, ch: Channels, obs: Optional[FaceObs],
               ipd: Optional[float], fps: float, det_ms: float,
               affect: Optional["AffectState"] = None,
               scale: float = 2.0) -> np.ndarray:
    """Readout strip under the video.

    Every dimension derives from `scale` rather than being a literal, because
    the panel is composed at FULL frame width and the whole view is downscaled
    afterwards to fit a screen -- at a 3840-wide capture that is a factor of
    0.42, so anything sized to look right here arrives on screen at less than
    half of it. Scaling the panel up front is what survives that.
    """
    K = scale

    def q(v):
        return int(round(v * K))

    P = np.full((q(190), width, 3), 22, np.uint8)
    f = cv2.FONT_HERSHEY_SIMPLEX
    s = 0.46 * K
    lw = max(1, int(round(K)))
    col = (215, 215, 215)

    def txt(x, y, t, c=col, sc=s):
        cv2.putText(P, t, (q(x), q(y)), f, sc, c, lw, cv2.LINE_AA)

    txt(10, 22, "EXPRESSION CHANNELS", (150, 150, 150), 0.5 * K)

    txt(10, 52, f"arousal {ch.arousal:+.2f}")
    _bar(P, q(130), q(40), q(200), q(14), ch.arousal, (90, 160, 255),
         signed=True, lw=lw)
    txt(10, 78, f"valence {ch.valence:+.2f}")
    _bar(P, q(130), q(66), q(200), q(14), ch.valence, (120, 220, 130),
         signed=True, lw=lw)
    txt(10, 104, f"change  {ch.change:.2f}/s")
    _bar(P, q(130), q(92), q(200), q(14), min(ch.change / 2.0, 1.0),
         (200, 170, 90), lw=lw)

    x2 = 360
    if ch.pos is not None:
        txt(x2, 52, f"head  X{ch.pos[0]:+.2f} Y{ch.pos[1]:+.2f} Z{ch.pos[2]:+.2f} m")
        txt(x2, 78, f"dist  {ch.distance:.2f} m   approach {ch.approach:+.2f} m/s")
    else:
        txt(x2, 52, "head  -- no 3-D fix --", (120, 120, 200))

    attending = obs is not None and Channels.attending(obs)
    txt(x2, 104, f"dwell {ch.dwell:5.1f}s",
        (120, 230, 140) if attending else (140, 140, 140))
    txt(x2 + 130, 104, "ATTENDING" if attending else
        ("absent %.1fs" % ch.absent if obs is None else "looking away"),
        (120, 230, 140) if attending else (140, 140, 140))

    # IPD is the honest check on whether the 3-D numbers mean anything at all
    if ipd is not None:
        ok = 0.050 <= ipd <= 0.080
        txt(10, 140, f"IPD {ipd*1000:5.1f} mm", (120, 230, 140) if ok else (90, 90, 235))
        txt(130, 140,
            "plausible" if ok else "IMPLAUSIBLE - stereo scale is off at face range",
            (120, 230, 140) if ok else (90, 90, 235))
    else:
        txt(10, 140, "IPD --", (130, 130, 130))

    # derived state, drawn large: this is the output the state machine consumes,
    # so it should be readable from across the room rather than squinted at
    if affect is not None:
        st = affect.state
        txt(x2, 140, "state", (150, 150, 150))
        cv2.putText(P, st.upper(), (q(x2 + 62), q(142)), f, 0.72 * K,
                    STATE_BGR.get(st, (200, 200, 200)), lw + 1, cv2.LINE_AA)
        ab, vb = affect.baseline
        txt(x2 + 250, 140, f"base a{ab:+.2f} v{vb:+.2f}", (120, 120, 120), 0.40 * K)

    txt(10, 168, f"{fps:5.1f} fps   detect {det_ms:5.1f} ms", (140, 140, 140))
    txt(x2, 168, "q quit   r reset channels", (140, 140, 140))
    return P


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0",
                    help="camera index, or a video/image path (avoids camera "
                         "contention with the closed loop)")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--flip-yaw", action="store_true")
    ap.add_argument("--flip-pitch", action="store_true")
    ap.add_argument("--config", default=None,
                    help=f"perception YAML (default: {DEFAULT_PERCEPTION_YAML.name})")
    ap.add_argument("--max-width", type=int, default=1600,
                    help="downscale the window to at most this many px wide; "
                         "0 keeps full resolution in a resizable window")
    ap.add_argument("--panel-scale", type=float, default=2.0,
                    help="size multiplier for the readout strip under the video")
    ap.add_argument("--crop", type=int, default=SEARCH_CROP,
                    help="square search-crop side in px; smaller sees further "
                         "but over a narrower field")
    args = ap.parse_args()

    # never get_perception_config(None) -- see DEFAULT_PERCEPTION_YAML
    cfg_path = args.config or (str(DEFAULT_PERCEPTION_YAML)
                               if DEFAULT_PERCEPTION_YAML.exists() else None)
    cfg = get_perception_config(cfg_path)
    if cfg_path is None:
        print("  ! perception YAML not found; using field defaults. "
              "units_to_meters may be wrong and every 3-D value with it.")
    print(f"  config {Path(cfg_path).name if cfg_path else '<defaults>'}: "
          f"units_to_meters={cfg.units_to_meters} "
          f"rotation={cfg.rotation_angle_deg} rotate180={cfg.camera_rotate_180}")
    try:
        calib = load_stereo_calibration(calib_dir=cfg.camera_calibration_path)
        print(f"  stereo calibration loaded from {cfg.camera_calibration_path}")
    except Exception as e:                                   # noqa: BLE001
        calib = None
        print(f"  ! no stereo calibration ({e}); 2-D only, no 3-D head position")

    src: object = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"cannot open source {args.source!r}")
    if isinstance(src, int):
        w, h = cfg.stereo_resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=2)

    ch = Channels()
    affect = AffectState()
    roi_l = roi_r = None
    writer = fh = None
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fh = open(args.csv, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["t", "det_l", "det_r", "x", "y", "z", "dist", "ipd",
                         "yaw", "pitch", "roll", "arousal", "valence", "change",
                         "dwell", "absent", "approach", "attending", "state",
                         "det_ms"])
        print(f"  logging to {args.csv}")

    # Mouse state lives in a dict because the callback runs on the GUI thread
    # and needs somewhere to put the click that the loop can read.
    ui = {"stop": False, "btn": (0, 0, 0, 0)}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        bx, by, bw, bh = ui["btn"]
        if bx <= x <= bx + bw and by <= y <= by + bh:
            ui["stop"] = True

    if not args.headless:
        # AUTOSIZE when scaling: the window is exactly the image, so a click is
        # trivially an image coordinate. NORMAL when not: the full-resolution
        # view is wider than any screen, so the window MUST be resizable to
        # reach the button at all -- and OpenCV maps clicks back into image
        # space for it. That mapping is backend-dependent, so if STOP stops
        # responding after a resize, that is the cause; drop --max-width 0.
        scaling = args.max_width > 0
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE if scaling else cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, on_mouse)
    print("  q or the STOP button quits, r resets the channel state\n")
    t0, frames, fps, last = time.time(), 0, 0.0, time.time()

    try:
        while True:
            tic = time.time()
            ok, frame = read_oriented(cap)
            if not ok:
                print("\n  source ended")
                break
            left, right = split_stereo_frame(frame)

            if roi_l is None:
                roi_l = Roi(left.shape[1], left.shape[0], args.crop)
                roi_r = Roi(right.shape[1], right.shape[0], args.crop)
                print(f"  eye {left.shape[1]}x{left.shape[0]}, "
                      f"search crop {roi_l.side}px "
                      f"(detector scale {192/roi_l.side:.2f} vs "
                      f"{192/left.shape[1]:.2f} uncropped)")

            d0 = time.time()
            fl, fr = pool.map(
                lambda a: detect_face(a[0], a[1], args.flip_yaw, args.flip_pitch),
                ((left, roi_l), (right, roi_r)))
            det_ms = (time.time() - d0) * 1000.0

            # --- stereo fusion: the step MediaPipe does not do for you --------
            pos = ipd = None
            if calib is not None and fl is not None and fr is not None:
                def tri(idx):
                    return triangulate_points(
                        fl.px[idx], fr.px[idx], calib,
                        units_to_m=cfg.units_to_meters,
                        rotation_angle_deg=cfg.rotation_angle_deg,
                        y_translation_m=cfg.y_translation_m,
                        coordinate_limits=FACE_LIMITS)

                pos = tri(NOSE_TIP)
                eye_l, eye_r = tri(IRIS_L), tri(IRIS_R)
                if eye_l is not None and eye_r is not None:
                    ipd = float(np.linalg.norm(eye_l - eye_r))

            obs = fl or fr
            now_t = time.time()
            ch.update(now_t, pos, obs)
            affect.update(now_t - t0, ch.arousal, ch.valence, obs is not None)

            if writer is not None:
                p = ch.pos if ch.pos is not None else (np.nan,) * 3
                writer.writerow([
                    f"{time.time() - t0:.3f}", int(fl is not None), int(fr is not None),
                    *[f"{v:.4f}" for v in p],
                    "" if ch.distance is None else f"{ch.distance:.4f}",
                    "" if ipd is None else f"{ipd:.4f}",
                    *(f"{v:.2f}" for v in ((obs.yaw, obs.pitch, obs.roll)
                                           if obs else (np.nan,) * 3)),
                    f"{ch.arousal:.3f}", f"{ch.valence:.3f}", f"{ch.change:.3f}",
                    f"{ch.dwell:.2f}", f"{ch.absent:.2f}", f"{ch.approach:.3f}",
                    int(obs is not None and Channels.attending(obs)),
                    affect.state, f"{det_ms:.1f}"])

            frames += 1
            if time.time() - last >= 0.5:
                fps = frames / (time.time() - last)
                frames, last = 0, time.time()
                print(f"\r  {'FACE' if obs else '----'} "
                      f"a={ch.arousal:+.2f} v={ch.valence:+.2f} "
                      f"dwell={ch.dwell:4.1f}s "
                      f"dist={'--' if ch.distance is None else f'{ch.distance:.2f}m'} "
                      f"ipd={'--' if ipd is None else f'{ipd*1000:.0f}mm'} "
                      f"{affect.state:11} "
                      f"{fps:4.1f}fps {det_ms:5.1f}ms", end="", flush=True)

            if not args.headless:
                view = np.hstack([draw_face(left, fl, "LEFT", roi_l),
                                  draw_face(right, fr, "RIGHT", roi_r)])
                view = np.vstack([view, draw_panel(view.shape[1], ch, obs,
                                                   ipd, fps, det_ms, affect,
                                                   scale=args.panel_scale)])
                # Two 1920-wide eyes make a 3840-wide view, which no laptop can
                # show at 1:1. Scale to fit, then draw the button on the scaled
                # image so its hit box matches where clicks actually land.
                if args.max_width > 0 and view.shape[1] > args.max_width:
                    s = args.max_width / view.shape[1]
                    view = cv2.resize(view, None, fx=s, fy=s,
                                      interpolation=cv2.INTER_AREA)
                ui["btn"] = draw_stop_button(view)
                cv2.imshow(WIN, view)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q") or ui["stop"]:
                    if ui["stop"]:
                        print("\n  -> STOP clicked")
                    break
                if k == ord("r"):
                    ch = Channels()
                    affect = AffectState()
                    roi_l.reset()
                    roi_r.reset()
                    print("\n  -> channels + affect baseline + ROI reset")

            if isinstance(src, str):                 # play files at ~30 fps
                time.sleep(max(0.0, 1 / 30 - (time.time() - tic)))
    except KeyboardInterrupt:
        pass
    finally:
        pool.shutdown()
        cap.release()
        cv2.destroyAllWindows()
        if fh:
            fh.close()
        print("\n  done.")


if __name__ == "__main__":
    main()
