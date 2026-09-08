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
import shutil
import subprocess
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
from ..perception.camera import read_oriented, set_camera_orientation
from ..perception.face import IRIS_L, IRIS_R, NOSE_TIP, Roi, detect_face
from ..perception.stereo import (load_stereo_calibration, split_stereo_frame,
                                 triangulate_points)

# Face-range triangulation box. triangulate_points CLIPS to coordinate_limits
# and indexes it unconditionally, and the project's configured limits are the
# tentacle's workspace -- a box a face never enters, which would silently pin
# every head position to a corner.
FACE_LIMITS = {a: {"clip_min": -3.0, "clip_max": 3.0} for a in "XYZ"}

GESTURE_BUFFER_S = 4.0      # history kept for the gesture detector

# Sound. A clip named after the primitive plays alongside it -- yes.m4a with the
# yes nod, no.m4a with the shake -- so adding one is dropping a file in, with no
# table or code change.
#
# afplay rather than a library: it ships with macOS and decodes m4a natively,
# which pygame's mixer does not ("Unrecognized audio format"). Converting the
# clips to wav would work too, but a subprocess that is guaranteed present beats
# a dependency plus a conversion step. If this ever needs to run off a Mac, that
# is the point to revisit.
AUDIO_DIR = Path(__file__).resolve().parents[1].parent / "assets" / "audio"
AUDIO_EXTS = (".m4a", ".wav", ".mp3", ".aiff", ".aif", ".ogg")


def sound_for(primitive: Optional[str]) -> Optional[Path]:
    """The clip that goes with a primitive, if there is one."""
    if not primitive or not AUDIO_DIR.is_dir():
        return None
    for ext in AUDIO_EXTS:
        p = AUDIO_DIR / f"{primitive}{ext}"
        if p.exists():
            return p
    return None


# =============================================================================
# what a state asks the body to do
# =============================================================================
@dataclass(frozen=True)
class BodyAction:
    """A state's motion, as data rather than a primitive name.

    `sound` is a path to a clip played alongside the primitive, or None. It was
    carried unused through v1 precisely so adding audio would not mean threading
    a new field through the worker.
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
HOLDS = {"grab_object": "release_object", "arched": "unarch"}

# Pause between repeats of a looping body, ON TOP of what execute_behavior
# already imposes: it ends every non-holding primitive with a reset to the
# calibrated pose and a hard 0.2 s sleep, and the mailbox poll adds 0.1 s. That
# floor of ~0.3 s was already visible as a stutter between cycles, so this is
# 0 rather than the 0.6 it started at -- sustained bodies are meant to read as
# continuous, and the residual gap is not ours to remove without changing
# execute_behavior, which orchestrate shares.
LOOP_GAP_S = 0.0


def body_for(state: str, states_rows: list[dict]) -> BodyAction:
    """Read a state's body out of the States table.

    The table says 'Play "slow_circle" motion primitive' in prose, because it is
    a design document a person edits. Parsing it here rather than maintaining a
    second mapping in code keeps the table the single source of truth -- the
    same reason the transitions are parsed rather than transcribed.
    """
    from ..control.primitives import MotionBehavior

    row = next((r for r in states_rows if r.get("State") == state), None)
    body = (row or {}).get("Body", "") or ""
    # Match against the primitives that actually EXIST rather than guessing at
    # the shape of a name. The first version looked for a lowercase token
    # containing an underscore, which silently missed `arched` -- SAD and ANGRY
    # resolved to no body at all and would have stood still. MotionBehavior is
    # the authority on what is a primitive, so a new one works here for free.
    known = {b.value.strip("<>") for b in MotionBehavior}
    tokens = [t.strip('"“”’\'.,') for t in body.split()]
    prim = next((t for t in tokens if t in known), None)
    hold = prim in HOLDS
    clip = sound_for(prim)
    return BodyAction(state=state, primitive=prim,
                      loop=state not in ONE_SHOT and not hold,
                      on_exit=HOLDS.get(prim),
                      sound=str(clip) if clip else None)


# =============================================================================
# motion, on its own thread
# =============================================================================
class MotionWorker:
    """Executes body actions without ever blocking perception.

    TWO slots, not one, because a state and a gesture are different kinds of
    thing. A state describes how things ARE, so the latest one wins and older
    ones are rightly discarded. A gesture is an acknowledgement of something a
    person DID, and dropping it means the robot ignored them.

    A single "latest wins" slot conflated the two, and the first motorised
    session showed the cost: YES was entered at 79.52 s and left at 80.12 s,
    exactly its 600 ms min duration, so post(YES) filled the slot and the
    post(NEUTRAL) that followed discarded it before the worker -- then inside an
    uninterruptible slow_circle -- had looked. The nod was detected, entered and
    exited without a single command reaching a motor.

    So gestures land in their own slot, are never overwritten by a state, and
    play at the first opportunity. The state slot keeps latest-wins.

    Primitives are blocking sleep-loops with no notion of cancellation, so a
    change cannot interrupt one mid-motion -- it takes effect at the next
    repeat. For the one-shot states that is correct anyway; for looping ones it
    bounds reaction time by the primitive's own length, which is the argument
    for keeping state bodies short.
    """

    def __init__(self, motor_controller=None, dry_run: bool = True):
        self.mc = motor_controller
        self.dry_run = dry_run
        self._lock = threading.Lock()
        self._pending_state: Optional[BodyAction] = None   # latest wins
        self._pending_event: Optional[BodyAction] = None   # never discarded
        self._resume: Optional[BodyAction] = None          # body to return to
        # Woken by post(). The two slots are checked without blocking, so
        # without this the idle worker only looks every 50 ms and a stop() can
        # beat a just-posted gesture to it -- which dropped YES from the
        # rehearsal. The old single mailbox blocked on get(), so posting always
        # woke it immediately; this restores that.
        self._wake = threading.Event()
        self._stop = threading.Event()
        # Set by post() and stop() to cut a primitive short. Primitives that
        # honour it check between command points and ease back to neutral, so a
        # state change lands in ~0.3 s instead of waiting out a play that can
        # run 25 s. Ones that do not honour it simply finish, as before.
        self._interrupt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.current: Optional[BodyAction] = None
        self.plays = 0
        self._audio: Optional[subprocess.Popen] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="mirror-motion")
        self._thread.start()

    @staticmethod
    def _is_event(action: BodyAction) -> bool:
        """A one-shot with nothing to undo: a gesture, not a state or a hold."""
        return action.primitive is not None and not action.loop and not action.on_exit

    def post(self, action: BodyAction) -> None:
        with self._lock:
            if self._is_event(action):
                self._pending_event = action
            else:
                self._pending_state = action
        self._interrupt.set()                  # cut short whatever is playing
        self._wake.set()                       # and look at the slots now

    def stop(self, timeout: float = 30.0, interrupt: bool = True) -> None:
        """Stop, and do not return until the body has actually stopped moving.

        Primitives are blocking sleep-loops with no cancellation, so setting the
        flag only prevents the NEXT one starting -- the current one runs to
        completion. slow_breathe is about 20 s. A short join therefore returns
        while the tentacle is still moving, which is merely untidy in a viewer
        and actively wrong anywhere that then measures position or drops the
        bus: net drift read mid-motion is meaningless, and disconnecting under
        a live primitive leaves commands in flight.

        The default is generous rather than tight for that reason. If it does
        time out, say so -- a silent overrun is the thing being guarded against.
        """
        # JOIN BEFORE RELEASING. This used to play the release first, from the
        # caller's thread, while the worker could still be mid-primitive -- so
        # the arch ramp and the unarch ramp interleaved, two threads issuing
        # conflicting positions down one bus. The bus lock serialises the
        # writes but cannot stop the COMMANDS alternating, which the static
        # check saw as a 570-tick step at 3.2M ticks/s.
        self._stop.set()
        self._wake.set()
        # interrupt=False lets the body play out, which is what a measurement
        # wants: mirror_rehearsal's static pass has to see the WHOLE primitive
        # to judge its range and rate, and cutting it short reported 23 commands
        # for motions that issue hundreds. Live operation wants the opposite.
        if interrupt:
            self._interrupt.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                print(f"\n  ! motion thread still running after {timeout:.0f}s; "
                      f"the body may still be moving")
        self._stop_sound()
        # Now that nothing else is driving the motors, undo a held pose.
        # _resume may hold the body underneath a gesture; it never has an
        # on_exit, so only `current` can be a hold.
        # Exiting arched or grabbed leaves tension on the tendons with nothing
        # holding it.
        if self.current is not None and self.current.on_exit:
            self._play(self.current.on_exit)

    def _play_sound(self, path: Optional[str]) -> None:
        """Start a clip alongside the motion. Fire and forget, never blocking.

        Sound has to begin WITH the movement, so this cannot be awaited on the
        motion thread -- a 1.6 s clip would delay the gesture by its own length.
        Any still-playing clip is stopped first: overlapping acknowledgements
        would be worse than a truncated one.
        """
        if not path:
            return
        player = shutil.which("afplay")
        if player is None:
            return
        self._stop_sound()
        try:
            self._audio = subprocess.Popen(
                [player, path], stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
        except Exception as exc:               # audio is never worth a crash
            print(f"\n  ! could not play {path}: {exc}")

    def _stop_sound(self) -> None:
        if self._audio is not None and self._audio.poll() is None:
            self._audio.terminate()
        self._audio = None

    def _play(self, primitive: str) -> bool:
        from ..control.primitives import MotionBehavior, execute_behavior
        behaviour = MotionBehavior.from_action_string(f"<{primitive}>")
        if behaviour is None:
            return False
        if self.dry_run or self.mc is None:
            time.sleep(0.4)                    # stand in for the motion
            return True
        try:
            execute_behavior(self.mc, behaviour, noise_scale=0.0,
                             should_stop=self._interrupt.is_set)
            return True
        except Exception as exc:               # a failed motion is not fatal
            print(f"\n  ! motion {primitive} failed: {exc}")
            time.sleep(0.2)
            return False

    def _take(self):
        """Next thing to play. Gestures jump the queue; states replace."""
        with self._lock:
            if self._pending_event is not None:
                act, self._pending_event = self._pending_event, None
                # Remember what to go back to, the way the state machine's
                # YES/NO rows return to the previous state. Any body that is
                # not itself a gesture, INCLUDING A HOLD: interrupting arched
                # and not restoring it would leave `current` as the gesture,
                # so its on_exit would be lost and nothing would ever unarch.
                if self.current is not None and not self._is_event(self.current):
                    self._resume = self.current
                return act, True
            if self._pending_state is not None:
                act, self._pending_state = self._pending_state, None
                self._resume = None
                return act, False
        return None, False

    def _run(self) -> None:
        while not self._stop.is_set():
            nxt, is_event = self._take()
            if nxt is not None:
                # Undo a hold before anything else. SAD and ANGRY arch and stay
                # arched; leaving has to unarch, or the next body plays from a
                # held pose. A gesture does not clear the hold -- it interrupts
                # and hands back.
                if not is_event and self.current is not None \
                        and self.current.on_exit:
                    self._play(self.current.on_exit)
                self.current, self.plays = nxt, 0
            elif self.current is not None and not self.current.loop:
                # a one-shot has had its turn; go back to the body underneath it
                if self._resume is not None:
                    self.current, self._resume, self.plays = self._resume, None, 0
                else:
                    self._wake.wait(0.1)
                    self._wake.clear()
                    continue
            elif self.current is None:
                self._wake.wait(0.1)
                self._wake.clear()
                continue

            act = self.current
            if act is None or act.primitive is None:
                self._wake.wait(0.1)
                self._wake.clear()
                continue
            self._interrupt.clear()            # fresh play, fresh flag
            self.plays += 1
            if self.plays == 1:                # once per entry, not per repeat
                self._play_sound(act.sound)
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
WIN = "shoggoth mirror"


def run_mirror(tables: Path, source="0", use_motors: bool = False,
               csv_path: Optional[Path] = None, config: Optional[str] = None,
               crop: int = 0, flip_yaw: bool = False,
               flip_pitch: bool = False, view: bool = False,
               view_eyes: str = "left", view_width: int = 1600,
               panel_scale: float = 2.0) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from ..configs import get_perception_config
    from ..perception.face import SEARCH_CROP
    from ..perception.overlay import (draw_face, draw_mirror_strip,
                                      draw_panel, draw_stop_button,
                                      min_panel_width)

    pkg = Path(__file__).resolve().parents[1]
    yaml = pkg / "configs" / "default_perception.yaml"
    # never get_perception_config(None): it returns pydantic defaults and
    # silently ignores the YAML, which puts units_to_meters at 0.05 instead of
    # 1.0 and makes every triangulated distance 20x too small.
    cfg_path = config or (str(yaml) if yaml.exists() else None)
    cfg = get_perception_config(cfg_path)
    # camera.read_oriented resolves camera_rotate_180 from the DEFAULT yaml,
    # independently of whatever config the caller loaded. Pass --config with a
    # different orientation and the frame rotation would disagree with the
    # triangulation -- and a 180 degree rotation swaps which half of the
    # side-by-side frame belongs to which physical camera, so the stereo
    # baseline inverts and every 3-D estimate is geometrically wrong while
    # still looking plausible. Bind them together explicitly.
    set_camera_orientation(cfg.camera_rotate_180)
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
        from ..configs import get_hardware_config
        from ..hardware.motors import MotorController

        # MotorController() with no config builds HardwareConfig() from pydantic
        # defaults, NOT from the YAML -- the same trap as get_perception_config
        # with no argument. It matters more here: the default tick_sign is +1
        # against the YAML's -1, so every commanded direction would be reversed,
        # and the range guards would not notice because the positions are legal,
        # merely mirrored. The default port is also empty.
        hw_yaml = pkg / "configs" / "default_hardware.yaml"
        hw = get_hardware_config(str(hw_yaml) if hw_yaml.exists() else None)
        if not hw.port:
            raise SystemExit(
                "no motor port configured. Expected it in "
                f"{hw_yaml}; refusing to guess.")
        mc = MotorController(hw)
        print(f"  motors: port {hw.port} tick_sign {hw.tick_sign} "
              f"settle {hw.motor_settle_time}s")
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

    # The viewer runs on THIS thread. cv2's window pump is main-thread-only on
    # macOS, and the perception loop is already the main thread -- only motors
    # are on a worker -- so the window is safe here and nowhere else. Drawing
    # costs a copy plus 468 circles per eye at full capture width, which is why
    # one eye is the default: two views of the same face tell a viewer nothing
    # extra and double the bill.
    ui = {"stop": False, "btn": (0, 0, 0, 0)}
    min_w = min_panel_width(panel_scale)
    if view:
        def on_mouse(event, x, y, flags, _):
            if event == cv2.EVENT_LBUTTONDOWN:
                bx, by, bw, bh = ui["btn"]
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    ui["stop"] = True
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, on_mouse)
        print(f"  view: ON ({view_eyes} eye, max {view_width}px, "
              f"min {min_w}px) -- q or STOP to quit cleanly")
    print("  ctrl-C to stop\n")
    vfps, vframes, vlast = 0.0, 0, time.time()

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

            if view:
                vframes += 1
                if time.time() - vlast >= 0.5:
                    vfps = vframes / (time.time() - vlast)
                    vframes, vlast = 0, time.time()
                if view_eyes == "both":
                    canvas = np.hstack([draw_face(left, fl, "LEFT", roi_l),
                                        draw_face(right, fr, "RIGHT", roi_r)])
                elif view_eyes == "right":
                    canvas = draw_face(right, fr, "RIGHT", roi_r)
                else:
                    canvas = draw_face(left, fl, "LEFT", roi_l)
                # Upscale a narrow capture BEFORE composing. putText clips
                # without complaint, so a composition narrower than the panels
                # need loses the input chips and the playing primitive while
                # still looking like a working display.
                if canvas.shape[1] < min_w:
                    sc = min_w / canvas.shape[1]
                    canvas = cv2.resize(canvas, None, fx=sc, fy=sc,
                                        interpolation=cv2.INTER_LINEAR)
                # The strip is fed the SAME vector and row the machine just
                # stepped on, so the screen cannot claim a transition the
                # machine did not make.
                cur = worker.current
                canvas = np.vstack([
                    canvas,
                    draw_panel(canvas.shape[1], ch, obs, ipd, vfps, det_ms,
                               affect, scale=panel_scale),
                    draw_mirror_strip(canvas.shape[1], state,
                                      cur.primitive if cur else None, v,
                                      fired.row if fired else None, emo,
                                      scale=panel_scale)])
                if view_width > 0 and canvas.shape[1] > view_width:
                    sc = view_width / canvas.shape[1]
                    canvas = cv2.resize(canvas, None, fx=sc, fy=sc,
                                        interpolation=cv2.INTER_AREA)
                # Drawn AFTER the downscale so the hit box is in the same
                # coordinates the mouse callback reports clicks in.
                ui["btn"] = draw_stop_button(canvas)
                cv2.imshow(WIN, canvas)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q") or ui["stop"]:
                    # break, never sys.exit: the finally block is what parks
                    # the motors and disconnects the bus.
                    print(f"\n  stopping ({'STOP clicked' if ui['stop'] else 'q'})")
                    break

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
        if view:
            cv2.destroyAllWindows()
        if fh:
            fh.close()
        if mc is not None:
            try:
                mc.reset_to_calibrated_positions()
            finally:
                mc.disconnect()
            print("  motors returned to calibrated position and disconnected")
        print("  done.")
