"""Interactive motion playground -- live tentacle in MuJoCo (STUB).

DESIGN/RESEARCH (see NEXT_PHASE_EXPLORATION.md). Reuses the headless generator
from motion_playground.py; drives the simulated tentacle's tendons directly
(open-loop, no RL policy) so you can *watch* each mood and judge it by eye.

    >>> mjpython exploration/motion_viewer.py        # macOS needs mjpython, not python

A live rolling plot of the three commanded tendon lengths is drawn as an overlay
*inside* the MuJoCo window (beside the tentacle; red line = shortening floor ->
watch for clipping). It's in-scene geometry, so no second GUI window is needed
(cv2/tk windows don't play nice with GLFW under mjpython here). If you don't see
it, orbit/zoom the camera; tweak OVERLAY_ORIGIN below to reposition.

Keys (focus the viewer window):
    1..6   switch mood   (sleepy content curious alert wary excited)
    SPACE  pause / resume
    [ / ]  scale all amplitudes down / up (probe the saturation finding)

This is the arbiter for the headless findings:
  #1 spatial similarity  -- do the moods actually look different?
  #2 saturation          -- do high-arousal moods clip / flatten?
  #3 'snappy' == squarish -- does excited read as lively or mechanical?
"""

import sys
import time
from collections import deque
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "exploration"))
from motion_playground import (  # noqa: E402  (reuse the headless generator)
    MOODS, affect_to_params, generate_trajectory, cursor_to_tendons,
    ACT_LOW, ACT_HIGH,
)

XML_PATH = REPO / "assets" / "simulation" / "tentacle.xml"
LOOP_SECONDS = 60.0                       # regenerated per mood, looped
LIVE_PLOT = True                          # rolling tendon-length overlay in the sim
OVERLAY_ORIGIN = (0.15, 0.0, -0.10)       # left-bottom corner of the plot (world m)
OVERLAY_W, OVERLAY_H = 0.30, 0.25         # plot width (time) x height (length), m

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)
DT = model.opt.timestep                   # 0.005 s
FPS = int(round(1.0 / DT))                # generate signals at the sim rate

mood_names = list(MOODS.keys())

# shared state mutated by the key callback (runs on the viewer thread)
state = {"mood_idx": 2, "paused": False, "amp_scale": 1.0, "dirty": True}


class LiveOverlay:
    """Rolling tendon-length plot drawn as line geometry INSIDE the MuJoCo scene.

    No second GUI window (cv2/tk fail under mjpython here) -- it populates
    viewer.user_scn with line geoms each throttled tick, so it renders through
    the viewer that's already running. The plot is a flat panel in the y=const
    plane: X axis = time (scrolls), Z axis = tendon length in [ACT_LOW, ACT_HIGH].
    """

    COLORS = [(0.20, 0.45, 1.0, 1.0),   # tendon 1  blue
              (1.00, 0.55, 0.15, 1.0),   # tendon 2  orange
              (0.25, 0.80, 0.35, 1.0)]   # tendon 3  green

    def __init__(self, n: int = 100, hz: int = 20):
        self.n = n
        self.every = max(1, int(round((1.0 / DT) / hz)))
        self.buf = [deque(maxlen=n) for _ in range(3)]
        self.o = np.array(OVERLAY_ORIGIN, float)
        self.W, self.H = OVERLAY_W, OVERLAY_H
        self._k = 0

    def push(self, L) -> None:
        for i in range(3):
            self.buf[i].append(float(L[i]))

    def _pt(self, j: int, length: float) -> np.ndarray:
        x = self.o[0] + (j / (self.n - 1)) * self.W
        frac = (np.clip(length, ACT_LOW, ACT_HIGH) - ACT_LOW) / (ACT_HIGH - ACT_LOW)
        return np.array([x, self.o[1], self.o[2] + frac * self.H])

    def draw(self, viewer) -> None:
        self._k += 1
        if self._k % self.every:
            return
        scn = viewer.user_scn
        g = 0

        def line(p0, p1, rgba, w=2.5):
            nonlocal g
            if g >= scn.maxgeom:
                return
            gm = scn.geoms[g]
            mujoco.mjv_initGeom(gm, mujoco.mjtGeom.mjGEOM_LINE,
                                np.zeros(3), np.zeros(3), np.zeros(9),
                                np.asarray(rgba, np.float32))
            mujoco.mjv_connector(gm, mujoco.mjtGeom.mjGEOM_LINE, w,
                                 p0.astype(np.float64), p1.astype(np.float64))
            g += 1

        with viewer.lock():
            o, W, H = self.o, self.W, self.H
            c00, c10 = o, o + [W, 0, 0]
            c01, c11 = o + [0, 0, H], o + [W, 0, H]
            for a, b in [(c01, c11), (c00, c01), (c10, c11)]:      # frame (no bottom)
                line(np.asarray(a), np.asarray(b), (0.5, 0.5, 0.5, 0.7), 1.5)
            line(np.asarray(c00), np.asarray(c10), (1.0, 0.25, 0.25, 0.9), 2.5)  # floor
            for i in range(3):
                pts = list(self.buf[i])
                for j in range(1, len(pts)):
                    line(self._pt(j - 1, pts[j - 1]), self._pt(j, pts[j]),
                         self.COLORS[i])
            scn.ngeom = g


def build_signals(mood_idx: int, amp_scale: float) -> dict:
    """Generate a long looped (c_x, c_y, a) stream for the current mood."""
    aff = MOODS[mood_names[mood_idx]]
    p = affect_to_params(aff)
    p.reach *= amp_scale                  # [ / ] probe for the saturation finding
    return generate_trajectory(p, duration=LOOP_SECONDS, fps=FPS, seed=7)


def key_callback(keycode: int) -> None:
    ch = chr(keycode) if 0 <= keycode < 0x110000 else ""
    if ch in "123456":
        i = int(ch) - 1
        if i < len(mood_names):
            state["mood_idx"] = i
            state["dirty"] = True
            print(f"  -> mood: {mood_names[i]}")
    elif keycode == ord(" "):
        state["paused"] = not state["paused"]
        print("  -> paused" if state["paused"] else "  -> resumed")
    elif ch == "]":
        state["amp_scale"] = min(1.5, state["amp_scale"] + 0.1)
        state["dirty"] = True
        print(f"  -> amp_scale {state['amp_scale']:.1f}")
    elif ch == "[":
        state["amp_scale"] = max(0.2, state["amp_scale"] - 0.1)
        state["dirty"] = True
        print(f"  -> amp_scale {state['amp_scale']:.1f}")


def main() -> None:
    print(__doc__)
    print(f"  moods: {', '.join(f'{i+1}={n}' for i, n in enumerate(mood_names))}\n")

    sig = build_signals(state["mood_idx"], state["amp_scale"])
    n = len(sig["t"])
    k = 0

    # settle at baseline before driving
    data.ctrl[:] = cursor_to_tendons(0.0, 0.0, 0.0)
    for _ in range(100):
        mujoco.mj_step(model, data)

    overlay = LiveOverlay() if LIVE_PLOT else None

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
    ) as viewer:
        print(f"  -> mood: {mood_names[state['mood_idx']]}")
        while viewer.is_running():
            tic = time.perf_counter()

            if state["dirty"]:
                sig = build_signals(state["mood_idx"], state["amp_scale"])
                n = len(sig["t"])
                k = 0
                state["dirty"] = False

            if not state["paused"]:
                data.ctrl[:] = cursor_to_tendons(
                    float(sig["cx"][k]), float(sig["cy"][k]), float(sig["a"][k])
                )
                mujoco.mj_step(model, data)
                k = (k + 1) % n
                if overlay is not None:
                    overlay.push(data.ctrl)
                    try:
                        overlay.draw(viewer)
                    except Exception as e:  # noqa: BLE001 -- never let it kill the sim
                        print(f"  (overlay stopped: {e})")
                        overlay = None

            viewer.sync()

            # keep it near real-time
            dt_left = DT - (time.perf_counter() - tic)
            if dt_left > 0:
                time.sleep(dt_left)


if __name__ == "__main__":
    main()
