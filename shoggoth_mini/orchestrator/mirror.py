"""orchestrate_mirror — the tentacle acting on the mirror state machine.

Distinct from the author's `orchestrate`, which is an LLM in a voice loop that
calls motion primitives as tools. This one has no model in it at all: stereo
cameras produce inputs, the state machine in shoggoth-mirror-state/ decides what
state the robot is in, and each state has a body. Perception -> state -> motion,
nothing else. Voice is deferred.

    python -m shoggoth_mini orchestrate-mirror                    # no motors
    python -m shoggoth_mini orchestrate-mirror --motors --csv s.csv

MOTORS ARE OFF BY DEFAULT. The whole loop, the state machine and the logging run
without ever connecting to the bus, which is the mode to develop in: you can
watch the machine react to a real person with nothing able to move.

THE LOG IS THE POINT. It writes the same CSV columns face_probe does, plus the
state and the row that fired. So a session that behaved oddly can be replayed
offline through the very tools that were used to design the machine:

    python tools/plot_face_csv.py    session.csv
    python tools/fsm_check.py shoggoth-mirror-state/ --replay session.csv --plot p.png

That is the reason perception, affect and the state machine were pulled into the
package: the robot and the tools that explain it run the same code, so a replay
is evidence rather than a second opinion.

Two things worth understanding before trusting what it does:

GESTURE LATENCY IS DELIBERATE. detect_head_gestures uses a window CENTRED on the
sample it scores -- it looks half a window into the future, which offline is free
and live is impossible. Rather than write a second, causal detector that would
drift from the offline one, this evaluates the newest FULLY WINDOWED sample,
which is win/2 = 0.5 s old. Identical arithmetic, known lag. The state machine
already waits 500 ms for a nod, so the lag is additive to a delay that exists
anyway, and a nod is acknowledged about a second after it starts.

THE AFFECT BASELINE IS RUNNING, NOT WHOLE-TAKE. AffectState uses a median over
the last 20 s, because that is all a live system can have. Replaying the same
session offline with plot_face_csv's default recomputes from the whole take and
will disagree on 10-25% of frames. Neither is wrong; the live one is what the
robot acted on, which is why the plotter shades by the logged column.
"""

from __future__ import annotations

import csv
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..affect import (Channels, AffectState, attending, detect_head_gestures,
                      STATE_NAMES)
from ..affect.config import GESTURE_WINDOW_S
from ..affect.fsm import EMOTIONS, Machine, load_tables, min_dur
from ..perception.camera import read_oriented
from ..perception.face import IRIS_L, IRIS_R, NOSE_TIP, Roi, detect_face
from ..perception.stereo import (load_stereo_calibration, split_stereo_frame,
                                 triangulate_points)

# Face-range triangulation box. triangulate_points CLIPS to coordinate_limits
# and indexes it unconditionally, and the project's configured limits are the
# tentacle's workspace -- a box a face never enters, which would silently pin
# every head position to a corner.
FACE_LIMITS = {a: {"clip_min": -3.0, "clip_max": 3.0} for a in "XYZ"}

GESTURE_BUFFER_S = 4.0      # history kept for the gesture detector


# =============================================================================
# what a state asks the body to do
# =============================================================================
@dataclass(frozen=True)
class BodyAction:
    """A state's motion, as data rather than a primitive name.

    `sound` is unused in v1 and present on purpose: yes/no acknowledgements are
    meant to gain audio, and threading a second field through the worker later
    is more disruptive than carrying an unused one now.
    """
    state: str
    primitive: Optional[str]
    loop: bool = True
    on_exit: Optional[str] = None
    sound: Optional[str] = None


# States that punctuate rather than persist: they play once and hand the body
# back. Everything else loops for as long as the machine stays in it.
ONE_SHOT = {"YES", "NO"}

# Primitives that END somewhere rather than returning to neutral, and what
# undoes them. This is a property of the PRIMITIVE, not of the state that asks
# for it -- execute_behavior already says the same thing by setting
# reset_after_sequence = False for grab. Deriving it here means a state mapped
# to grab later gets the right treatment without anyone remembering to say so.
#
# Looping a hold would be actively dangerous: it re-curls from an already
# curled pose with nothing ever releasing, and char_primitive_sweep's own
# safety note calls grab "38.5 mm of cable out of two motors at once, fast,
# which is exactly how wire comes off a roller when there is no tension on it".
HOLDS = {"grab_object": "release_object"}

# Pause between repeats of a looping body. Without it the worker re-fires as
# soon as its mailbox poll times out, 0.1 s later, which reads as relentless
# rather than alive.
LOOP_GAP_S = 0.6


def body_for(state: str, states_rows: list[dict]) -> BodyAction:
    """Read a state's body out of the States table.

    The table says 'Play "slow_circle" motion primitive' in prose, because it is
    a design document a person edits. Parsing it here rather than maintaining a
    second mapping in code keeps the table the single source of truth -- the
    same reason the transitions are parsed rather than transcribed.
    """
    row = next((r for r in states_rows if r.get("State") == state), None)
    body = (row or {}).get("Body", "") or ""
    tokens = [t.strip('"“”’\' ') for t in body.split()]
    prim = next((t for t in tokens if t and t.islower() and "_" in t or
                 t in {"yes", "no", "shake", "circle"}), None)
    hold = prim in HOLDS
    return BodyAction(state=state, primitive=prim,
                      loop=state not in ONE_SHOT and not hold,
                      on_exit=HOLDS.get(prim))


# =============================================================================
# motion, on its own thread
# =============================================================================
class MotionWorker:
    """Executes body actions without ever blocking perception.

    A one-slot mailbox rather than a queue: if the state changed twice while a
    primitive was playing, only the current state matters, and a backlog of
    stale intents is worse than dropping them.

    Primitives are blocking sleep-loops with no notion of cancellation, so a
    change cannot interrupt one mid-motion -- it takes effect at the next
    repeat. For the one-shot states that is correct anyway; for looping ones it
    bounds reaction time by the primitive's own length, which is the argument
    for keeping state bodies short.
    """

    def __init__(self, motor_controller=None, dry_run: bool = True):
        self.mc = motor_controller
        self.dry_run = dry_run
        self._mail: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.current: Optional[BodyAction] = None
        self.plays = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="mirror-motion")
        self._thread.start()

    def post(self, action: BodyAction) -> None:
        try:                                   # latest wins
            self._mail.get_nowait()
        except queue.Empty:
            pass
        self._mail.put_nowait(action)

    def stop(self) -> None:
        # Release a held grip before shutting down. Exiting with the tentacle
        # curled leaves tension on the tendons with nothing driving them.
        if self.current is not None and self.current.on_exit:
            self._play(self.current.on_exit)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _play(self, primitive: str) -> bool:
        from ..control.primitives import MotionBehavior, execute_behavior
        behaviour = MotionBehavior.from_action_string(f"<{primitive}>")
        if behaviour is None:
            return False
        if self.dry_run or self.mc is None:
            time.sleep(0.4)                    # stand in for the motion
            return True
        try:
            execute_behavior(self.mc, behaviour, noise_scale=0.0)
            return True
        except Exception as exc:               # a failed motion is not fatal
            print(f"\n  ! motion {primitive} failed: {exc}")
            time.sleep(0.2)
            return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                nxt = self._mail.get(timeout=0.1)
                # Undo a hold before doing anything else. SAD and ANGRY grab and
                # stay grabbed; leaving them has to let go, or the next state's
                # motion plays from a curled pose.
                if self.current is not None and self.current.on_exit:
                    self._play(self.current.on_exit)
                self.current, self.plays = nxt, 0
            except queue.Empty:
                if self.current is None or not self.current.loop:
                    continue
            act = self.current
            if act is None or act.primitive is None:
                time.sleep(0.05)
                continue
            self.plays += 1
            self._play(act.primitive)
            if act.loop:
                time.sleep(LOOP_GAP_S)


# =============================================================================
# perception -> inputs
# =============================================================================
class GestureWindow:
    """Streaming nod/shake, using the offline detector on a trailing buffer.

    detect_head_gestures scores a sample using a window CENTRED on it, so the
    newest sample can never be scored -- half its window has not happened yet.
    Rather than write a causal variant that would drift from the offline one,
    this keeps a few seconds of history, runs the same function over it, and
    reads the newest sample that has a complete window. The answer is therefore
    identical to what a later replay will produce, delayed by win/2.
    """

    def __init__(self, win: float = GESTURE_WINDOW_S,
                 buffer_s: float = GESTURE_BUFFER_S):
        self.win, self.buffer_s = win, buffer_s
        self.t: list[float] = []
        self.yaw: list[float] = []
        self.pitch: list[float] = []

    @property
    def lag_s(self) -> float:
        return self.win / 2.0

    def update(self, t: float, yaw: float, pitch: float) -> tuple[int, int]:
        self.t.append(t)
        self.yaw.append(yaw if yaw is not None else np.nan)
        self.pitch.append(pitch if pitch is not None else np.nan)
        while self.t and t - self.t[0] > self.buffer_s:
            self.t.pop(0); self.yaw.pop(0); self.pitch.pop(0)
        if len(self.t) < 12:
            return 0, 0
        ta = np.asarray(self.t)
        nod, shake = detect_head_gestures(ta, np.asarray(self.yaw),
                                          np.asarray(self.pitch))
        i = int(np.searchsorted(ta, t - self.lag_s, "right")) - 1
        if i < 0:
            return 0, 0
        return int(nod[i]), int(shake[i])


CSV_COLUMNS = ["t", "det_l", "det_r", "x", "y", "z", "dist", "ipd",
               "yaw", "pitch", "roll", "arousal", "valence", "change",
               "dwell", "absent", "approach", "attending", "state",
               "det_ms", "fsm_state", "fsm_row"]


# =============================================================================
def run_mirror(tables: Path, source="0", use_motors: bool = False,
               csv_path: Optional[Path] = None, config: Optional[str] = None,
               crop: int = 0, flip_yaw: bool = False,
               flip_pitch: bool = False) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from ..configs import get_perception_config
    from ..perception.face import SEARCH_CROP

    pkg = Path(__file__).resolve().parents[1]
    yaml = pkg / "configs" / "default_perception.yaml"
    # never get_perception_config(None): it returns pydantic defaults and
    # silently ignores the YAML, which puts units_to_meters at 0.05 instead of
    # 1.0 and makes every triangulated distance 20x too small.
    cfg_path = config or (str(yaml) if yaml.exists() else None)
    cfg = get_perception_config(cfg_path)
    print(f"  perception: units_to_meters={cfg.units_to_meters} "
          f"rotation={cfg.rotation_angle_deg} rotate180={cfg.camera_rotate_180}")

    states, states_rows, timing, transitions = None, None, None, None
    inputs, aliases, states, states_rows, timing, transitions = load_tables(tables)
    print(f"  machine: {len(states)} states, {len(transitions)} rows "
          f"from {tables}")

    try:
        calib = load_stereo_calibration(calib_dir=cfg.camera_calibration_path)
    except Exception as exc:                                   # noqa: BLE001
        calib = None
        print(f"  ! no stereo calibration ({exc}); 2-D only, no head position")

    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {source!r}")
    if isinstance(src, int):
        w, h = cfg.stereo_resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    mc = None
    if use_motors:
        from ..hardware.motors import MotorController
        mc = MotorController()
        mc.connect()
        print("  motors: CONNECTED")
    else:
        print("  motors: OFF (--motors to enable)")

    worker = MotionWorker(mc, dry_run=not use_motors)
    worker.start()
    machine = Machine(states, transitions, timing, states[0])
    ch, affect, gestures = Channels(), AffectState(), GestureWindow()
    roi_l = roi_r = None
    pool = ThreadPoolExecutor(max_workers=2)

    writer = fh = None
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(csv_path, "w", newline="")
        writer = csv.writer(fh); writer.writerow(CSV_COLUMNS)
        print(f"  logging to {csv_path}")

    print(f"  gesture lag {gestures.lag_s:.2f}s (centred window, evaluated on "
          f"the newest fully-windowed sample)")
    print("  ctrl-C to stop\n")

    t0 = time.time()
    last_state, last_print = None, 0.0
    # A file source delivers frames as fast as they decode, which collapses a
    # 30 s recording into a fraction of a second and makes every hold timer
    # ("Attention == 1 for > 1.5 s") unsatisfiable. Pace it to its own frame
    # rate so replaying a video exercises the machine the way a camera would.
    file_fps = 0.0
    if isinstance(src, str):
        file_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
        print(f"  file source: pacing to {file_fps:.0f} fps so hold timers mean "
              f"something")
    try:
        while True:
            tick = time.time()
            ok, frame = read_oriented(cap)
            if not ok:
                print("\n  source ended")
                break
            left, right = split_stereo_frame(frame)
            if roi_l is None:
                cw = crop or SEARCH_CROP
                roi_l = Roi(left.shape[1], left.shape[0], cw)
                roi_r = Roi(right.shape[1], right.shape[0], cw)

            d0 = time.time()
            fl, fr = pool.map(lambda a: detect_face(a[0], a[1], flip_yaw,
                                                    flip_pitch),
                              ((left, roi_l), (right, roi_r)))
            det_ms = (time.time() - d0) * 1000.0

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
                el, er = tri(IRIS_L), tri(IRIS_R)
                if el is not None and er is not None:
                    ipd = float(np.linalg.norm(el - er))

            obs = fl or fr
            now = time.time() - t0
            ch.update(time.time(), pos, obs)
            emo = affect.update(now, ch.arousal, ch.valence, obs is not None)
            nod, shake = gestures.update(
                now, obs.yaw if obs else np.nan, obs.pitch if obs else np.nan)

            face = int(obs is not None)
            att = int(face and attending(obs.yaw, obs.pitch))
            # No face means NO emotion asserted -- all five at zero, not
            # Neutral. 'calibrating' and 'unknown' are the same: an absence of
            # evidence rather than evidence of calm.
            v = {"Face": face, "Attention": att, "Yes": nod, "No": shake}
            v.update({e: 0 for e in EMOTIONS})
            label = emo.capitalize()
            if face and label in EMOTIONS:
                v[label] = 1

            state, fired, status = machine.step(now, v)
            if state != last_state:
                act = body_for(state, states_rows)
                worker.post(act)
                print(f"\n  {now:7.2f}s  {last_state or '-':9} -> {state:9} "
                      f"(row {fired.row if fired else '-'})  body="
                      f"{act.primitive or 'none'}{' loop' if act.loop else ''}")
                last_state = state

            if writer is not None:
                p = ch.pos if ch.pos is not None else (np.nan,) * 3
                writer.writerow([
                    f"{now:.3f}", int(fl is not None), int(fr is not None),
                    *[f"{x:.4f}" for x in p],
                    "" if ch.distance is None else f"{ch.distance:.4f}",
                    "" if ipd is None else f"{ipd:.4f}",
                    *(f"{x:.2f}" for x in ((obs.yaw, obs.pitch, obs.roll)
                                           if obs else (np.nan,) * 3)),
                    f"{ch.arousal:.3f}", f"{ch.valence:.3f}", f"{ch.change:.3f}",
                    f"{ch.dwell:.2f}", f"{ch.absent:.2f}", f"{ch.approach:.3f}",
                    att, emo, f"{det_ms:.1f}", state,
                    fired.row if fired else ""])

            if file_fps:
                slack = 1.0 / file_fps - (time.time() - tick)
                if slack > 0:
                    time.sleep(slack)

            if time.time() - t0 - last_print > 0.5:
                last_print = time.time() - t0
                print(f"\r  {state:9} face={face} att={att} emo={emo:11} "
                      f"nod={nod} shake={shake} {det_ms:5.1f}ms",
                      end="", flush=True)
    except KeyboardInterrupt:
        print("\n  stopping")
    finally:
        worker.stop()
        pool.shutdown()
        cap.release()
        if fh:
            fh.close()
        if mc is not None:
            try:
                mc.reset_to_calibrated_positions()
            finally:
                mc.disconnect()
            print("  motors returned to calibrated position and disconnected")
        print("  done.")
