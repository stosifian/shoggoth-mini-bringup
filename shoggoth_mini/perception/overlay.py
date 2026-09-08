"""Overlay drawing for the live face views.

Moved here verbatim from tools/face_probe.py so the probe and the mirror
orchestrator render from ONE implementation. They were always meant to show the
same thing; two copies would have drifted the moment either was tuned, and the
whole point of the affect package is that every driver agrees about what it is
looking at.

Nothing here holds state or decides anything -- it turns a frame plus the affect
channels into pixels. The panel is composed at FULL frame width and downscaled
by the caller afterwards, which is why every dimension derives from `scale`.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..affect import STATE_BGR, AffectState
from ..affect.fsm import EMOTIONS
from ..affect.channels import Channels
from .face import IRIS_L, IRIS_R, NOSE_TIP, FaceObs, Roi


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

    attending = obs is not None and ch.attending
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
# mirror strip -- the ROBOT's side, which face_probe has no concept of
# =============================================================================
INPUT_ORDER = ("Face", "Attention", "Yes", "No")


def draw_mirror_strip(width: int, state: str, primitive: Optional[str],
                      vec: dict, row: Optional[object], emotion: str,
                      scale: float = 2.0) -> np.ndarray:
    """The state machine's own readout: what it was told, and what it did.

    Deliberately shows the INPUT VECTOR rather than a summary. The whole design
    is a table driven by those bits, so a demo that shows a state changing
    without showing the bit that changed is asking to be taken on faith -- and
    when a transition looks wrong, this row is the first thing to read.

    `vec` is the same dict handed to Machine.step, so this cannot drift from
    what the machine actually saw.
    """
    K = scale

    def q(v):
        return int(round(v * K))

    P = np.full((q(96), width, 3), 28, np.uint8)
    f = cv2.FONT_HERSHEY_SIMPLEX
    lw = max(1, int(round(K)))

    def txt(x, y, t, c=(215, 215, 215), sc=0.46):
        cv2.putText(P, t, (q(x), q(y)), f, sc * K, c, lw, cv2.LINE_AA)

    txt(10, 22, "STATE MACHINE", (150, 150, 150), 0.42)
    cv2.putText(P, state.upper(), (q(10), q(60)), f, 0.86 * K,
                STATE_BGR.get(state.lower(), (235, 235, 235)), lw + 1,
                cv2.LINE_AA)
    txt(10, 84, f"row {row if row is not None else '-'}", (120, 120, 120), 0.40)

    # input bits as lit/unlit chips, in the table's own order
    x = 250
    txt(x, 22, "INPUTS", (150, 150, 150), 0.42)
    for i, name in enumerate(INPUT_ORDER):
        on = bool(vec.get(name))
        cx = x + i * 92
        cv2.rectangle(P, (q(cx), q(36)), (q(cx + 80), q(64)),
                      (70, 170, 90) if on else (48, 48, 52), -1)
        txt(cx + 8, 56, name[:9], (250, 250, 250) if on else (110, 110, 110), 0.40)
    # The Inputs table spells these "Emotion: Sad", but load_tables strips the
    # prefix -- the keys handed to Machine.step are bare ("Sad"). Filtering on
    # the table's spelling matched nothing and drew "none asserted" under an
    # ANGRY state. Take the list from the parser so it cannot disagree.
    lit = [e for e in EMOTIONS if vec.get(e)]
    txt(x, 84, f"emotion: {lit[0] if lit else '-- none asserted --'}"
               f"   (affect says {emotion})",
        (200, 200, 200) if lit else (120, 120, 120), 0.40)

    # what the body is actually doing
    x2 = 250 + 4 * 92 + 40
    txt(x2, 22, "BODY", (150, 150, 150), 0.42)
    cv2.putText(P, primitive or "none", (q(x2), q(60)), f, 0.62 * K,
                (120, 200, 255) if primitive else (110, 110, 110), lw + 1,
                cv2.LINE_AA)
    txt(x2, 84, "q quit   STOP click", (130, 130, 130), 0.40)
    return P


def min_panel_width(scale: float = 2.0) -> int:
    """Width below which the readouts start losing their right-hand column.

    Both panels lay content out in scale-relative units against a width they
    are simply HANDED, and cv2.putText clips silently -- no exception, no
    warning, the text is just not there. At a 320 px composition the state
    machine's input chips and the playing primitive both fell off the edge
    while everything looked fine. Compose narrower than this and the readout is
    lying by omission, so the caller should upscale to it instead.

    ~850 units is the strip's rightmost text ("q quit STOP click" at x2=658).
    """
    return int(round(850 * scale))
