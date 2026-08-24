"""PRIMITIVE SWEEP — actuate every designed motion and look for wind/unwind events.

Runs every motion-producing path in the codebase back to back, with the tendons
DETACHED, and reports what each one actually commands and what the motors actually
did. The question it answers is narrow and practical: does anything in the designed
behaviour set produce a sudden wind or unwind?

PHASE 1 is static and needs no hardware. It drives each primitive through a
recording stub that captures every Goal_Position it would write, and reports the
targets, the largest single-command delta, the cable travel, and the implied speed
from the primitive's own sleeps. This is the real safety net — it sees the whole
command set before anything moves, and it runs with the robot unplugged.

PHASE 2 executes them for real while a background thread samples all three motors,
then reports per primitive:
  * peak measured velocity
  * NET DRIFT — position at the end versus the start, after returning home. The
    code expects zero. A slipped roller or a lost turn shows up here and nowhere
    else, which is the entire reason this test exists.
  * commands that produced no motion (a ~3% silent drop rate was measured
    2026-08-18, independent of any of this)
  * any position leaving 0..4095

Primitives run smallest amplitude first, so the first surprise is the mildest.

SAFETY
  * TENDONS MUST BE DETACHED, not merely untied. GRAB at the upstream 0.7 pays
    38.5 mm of cable out of two motors at once, fast, which is exactly how wire
    comes off a roller when there is no tension on it.
  * --dry-run runs phase 1 only and moves nothing.
  * aborts if any motor leaves 0..4095 or if net drift exceeds --max-drift.
  * returns to the calibrated home between primitives so drift is judged per
    primitive rather than accumulating.

  python tools/char_primitive_sweep.py --dry-run
  python tools/char_primitive_sweep.py
  python tools/char_primitive_sweep.py --only grab,release
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.configs import get_hardware_config  # noqa: E402
from shoggoth_mini.hardware.motors import MotorController  # noqa: E402
from shoggoth_mini.common.constants import MOTOR_NAMES  # noqa: E402
from shoggoth_mini.control.primitives import MotionBehavior, execute_behavior  # noqa: E402
from shoggoth_mini.control.geometry import cursor_to_motor_positions  # noqa: E402
from char_common import Recorder, load, col  # noqa: E402

FIELDS = ["t", "phase", "motor", "present", "load"]
CMD_FIELDS = ["t", "phase", "motor", "target"]
PERIOD = 4096
TICKS_TO_MM = 0.11 / PERIOD * 1000

# Ordered by how hard each motion is on the mechanism, gentlest first — which is
# not the same as cursor magnitude. The tendon sweep reaches the LARGEST magnitude
# (0.25) but is rate-limited to 15 ticks per 10 ms loop, so it moves at ~1500
# ticks/s, a fifth of the servo ceiling. Grab is last because it is a single
# unramped command. RELEASE is the neutral pose, so it is a no-op opener.
ORDER = ["sweep", "release", "circle", "shake", "high_five", "yes", "no", "grab"]
BEHAVIOUR = {
    "sweep": "sweep",           # not a MotionBehavior: dispatched to perform_sweep
    "yes": MotionBehavior.YES, "no": MotionBehavior.NO,
    "shake": MotionBehavior.SHAKE, "circle": MotionBehavior.CIRCLE,
    "grab": MotionBehavior.GRAB, "release": MotionBehavior.RELEASE,
    "high_five": MotionBehavior.HIGH_FIVE,
}


class RecordingController:
    """Stand-in for MotorController that records commands instead of sending them.

    Primitives only need these four members, so phase 1 can drive the real
    primitive code — not a reimplementation of it — with no hardware attached.
    """

    def __init__(self, calib):
        self._calib = dict(calib)
        self._pos = dict(calib)     # simulated: perfect tracking of the last command
        self.commands = []          # (t, {motor: target})
        self.is_connected = True

    def get_calibration_data(self):
        return dict(self._calib)

    def get_position(self, motor):
        return self._pos[motor]

    def get_positions(self):
        return dict(self._pos)

    def set_positions(self, positions):
        cmd = {k: int(v) for k, v in positions.items()}
        self.commands.append((time.time(), cmd))
        self._pos.update(cmd)

    def set_position(self, motor, position):
        self.set_positions({motor: position})

    def reset_to_calibrated_positions(self):
        self.set_positions(self._calib)


class TeeController:
    """Passes every call through to the real controller and records the commands.

    Phase 2 otherwise cannot see what was commanded: the primitives call
    set_positions() on the controller directly, so the tool only ever observed the
    response. Without the command timeline there is no tracking error to plot.
    """

    def __init__(self, inner, on_command):
        self._inner = inner
        self._on_command = on_command

    def __getattr__(self, name):          # delegate everything not overridden
        return getattr(self._inner, name)

    def set_positions(self, positions):
        cmd = {k: int(v) for k, v in positions.items()}
        self._on_command(time.time(), cmd)
        self._inner.set_positions(positions)

    def set_position(self, motor, position):
        self.set_positions({motor: position})


def glide(controller, calib, target_cursor, step_ticks, hz=100.0):
    """Rate-limited move to a cursor position. Copied from tools/tendon_sweep.py
    so the sweep exercised here is the same motion that tool produces."""
    target, _ = cursor_to_motor_positions(
        cursor_pos=np.array(target_cursor, dtype=float), calibrated_ticks_map=calib
    )
    target = {m: int(target[m]) for m in MOTOR_NAMES}
    current = {m: controller.get_position(m) for m in MOTOR_NAMES}
    while True:
        deltas = {m: target[m] - current[m] for m in MOTOR_NAMES}
        if all(abs(d) <= step_ticks for d in deltas.values()):
            controller.set_positions(target)
            break
        for m in MOTOR_NAMES:
            current[m] += max(-step_ticks, min(step_ticks, deltas[m]))
        controller.set_positions(current)
        time.sleep(1.0 / hz)
    time.sleep(0.35)


def perform_sweep(controller, calib, magnitude=0.25, spokes=6, cycles=1,
                  step_ticks=15):
    """tendon_sweep --magnitude 0.25 --spokes 6 --cycles 1, inline.

    Runs first in the ordering: it is rate-limited to 15 ticks per 10 ms loop
    (~1500 ticks/s, a fifth of the servo ceiling), so it is by far the gentlest
    motion in the set despite reaching the largest cursor magnitude.
    """
    for _ in range(cycles):
        for i in range(spokes):
            a = 2 * np.pi * i / spokes
            glide(controller, calib,
                  (magnitude * np.cos(a), magnitude * np.sin(a)), step_ticks)
            glide(controller, calib, (0.0, 0.0), step_ticks)


def run_one(controller, name, calib, noise_scale):
    """Dispatch: the sweep is not a MotionBehavior, everything else is."""
    if name == "sweep":
        perform_sweep(controller, calib)
    else:
        execute_behavior(controller, BEHAVIOUR[name], noise_scale=noise_scale)


STATIC_FIELDS = ["primitive", "index", "t_offset", "motor", "target", "delta",
                 "cable_mm", "gap_s", "implied_ticks_s"]


def static_pass(names, calib, noise_scale, csv_path=None):
    """Phase 1: capture every command each primitive would issue."""
    rows = []
    srec = Recorder(csv_path, STATIC_FIELDS) if csv_path else None
    for name in names:
        rec = RecordingController(calib)
        t0 = time.time()
        run_one(rec, name, calib, noise_scale)
        wall = time.time() - t0

        if not rec.commands:
            rows.append(dict(name=name, n=0))
            continue

        per_motor = {m: [c[1][m] for c in rec.commands if m in c[1]]
                     for m in MOTOR_NAMES}
        max_step, max_travel, out = 0, 0, []
        for m in MOTOR_NAMES:
            seq = per_motor[m]
            if not seq:
                continue
            steps = [abs(b - a) for a, b in zip([calib[m]] + seq, seq)]
            max_step = max(max_step, max(steps))
            max_travel = max(max_travel, max(abs(v - calib[m]) for v in seq))
            out += [v for v in seq if not (0 <= v < PERIOD)]

        # Implied speed: the largest step divided by the gap the primitive leaves
        # before its next command. This is what the servo is asked to do, not what
        # it achieves — but a primitive that asks for more than the ~7600 ticks/s
        # ceiling is one whose timing assumptions are wrong.
        gaps = [b[0] - a[0] for a, b in zip(rec.commands, rec.commands[1:])]
        min_gap = min(gaps) if gaps else wall
        implied = max_step / min_gap if min_gap > 0 else float("inf")

        # Log every individual command, so the static pass leaves a record rather
        # than only terminal output.
        if srec is not None:
            t_start = rec.commands[0][0]
            prev = dict(calib)
            for i, (tc, cmd) in enumerate(rec.commands):
                gap = (rec.commands[i + 1][0] - tc) if i + 1 < len(rec.commands) else None
                for m, v in sorted(cmd.items()):
                    d = v - prev[m]
                    srec.log(primitive=name, index=i, t_offset=f"{tc - t_start:.4f}",
                             motor=m, target=v, delta=d,
                             cable_mm=f"{d * TICKS_TO_MM:.3f}",
                             gap_s=f"{gap:.4f}" if gap else "",
                             implied_ticks_s=f"{abs(d)/gap:.0f}" if gap else "")
                    prev[m] = v

        rows.append(dict(name=name, n=len(rec.commands), wall=wall,
                         max_step=max_step, max_travel=max_travel,
                         out_of_range=len(out), worst=max(out, key=abs) if out else None,
                         implied=implied,
                         lo={m: min(per_motor[m]) if per_motor[m] else None
                             for m in MOTOR_NAMES},
                         hi={m: max(per_motor[m]) if per_motor[m] else None
                             for m in MOTOR_NAMES}))
    if srec is not None:
        srec.close()
        print(f"\nstatic csv  -> {csv_path}")
    return rows


SERVO_CEILING = 7600        # ticks/s, measured in test C


def plot_static(csv_path, out_path, rows, calib):
    """Phase 1 figure: what the primitives ask for, before any hardware is involved."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load(csv_path)
    usable = [r for r in rows if r.get("n")]
    if not data or not usable:
        return

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 1])
    fig.suptitle("Primitive sweep, phase 1 — commanded trajectories (no hardware)",
                 fontsize=13)

    # panel 1: the commanded trajectory of every primitive, offset from home
    ax = fig.add_subplot(gs[0])
    colours = {"1": "tab:blue", "2": "tab:orange", "3": "tab:green"}
    x0 = 0.0
    for r in usable:
        sel = [d for d in data if d["primitive"] == r["name"]]
        for m in MOTOR_NAMES:
            s = [d for d in sel if d["motor"] == m]
            if len(s) < 1:
                continue
            t = col(s, "t_offset") + x0
            y = col(s, "target") - calib[m]
            ax.step(t, y, where="post", lw=1.3, color=colours[m],
                    label=f"motor {m}" if r is usable[0] else None)
        span = max((col(sel, "t_offset").max() if sel else 0), 0.05)
        ax.axvspan(x0, x0 + span, color="gray", alpha=.10)
        ax.text(x0 + span / 2, ax.get_ylim()[1], r["name"], fontsize=8,
                ha="center", va="top", rotation=90)
        x0 += span + 0.15
    ax.axhline(0, color="gray", lw=1)
    for lim, lab in ((-calib["1"], "0"), (PERIOD - 1 - calib["1"], "4095")):
        ax.axhline(lim, color="tab:red", lw=1, ls="--", alpha=.6)
    ax.set_ylabel("commanded offset from home (ticks)")
    ax.set_title("every Goal_Position each primitive writes; red lines are the "
                 "0 and 4095 bounds", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=.3)
    sec = ax.secondary_yaxis("right", functions=(lambda t: t * TICKS_TO_MM,
                                                 lambda s: s / TICKS_TO_MM))
    sec.set_ylabel("cable (mm)")

    # panel 2: travel and largest single step, per primitive
    ax = fig.add_subplot(gs[1])
    names = [r["name"] for r in usable]
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, [r["max_travel"] for r in usable], width=0.4,
           color="tab:purple", label="max travel from home")
    ax.bar(xs + 0.2, [r["max_step"] for r in usable], width=0.4,
           color="tab:cyan", label="largest single command")
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylabel("ticks")
    ax.set_title("how far each primitive goes, and how much it asks for at once",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")
    sec = ax.secondary_yaxis("right", functions=(lambda t: t * TICKS_TO_MM,
                                                 lambda s: s / TICKS_TO_MM))
    sec.set_ylabel("cable (mm)")

    # panel 3: implied rate against what the servo can actually do
    ax = fig.add_subplot(gs[2])
    imp = [min(r["implied"], 1e6) for r in usable]
    cols = ["tab:red" if v > SERVO_CEILING else "tab:green" for v in imp]
    ax.bar(xs, imp, color=cols)
    ax.axhline(SERVO_CEILING, color="black", lw=1.4, ls="--",
               label=f"{SERVO_CEILING} ticks/s measured ceiling")
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_yscale("log")
    ax.set_ylabel("implied rate (ticks/s, log)")
    ax.set_title("red bars ask for more than the servo can deliver — those "
                 "trajectories are set by servo dynamics, not by the waypoints",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    print(f"static plot -> {out_path}")


def print_static(rows):
    print(f"\n{'primitive':>11}{'cmds':>6}{'max step':>10}{'cable':>9}"
          f"{'travel':>8}{'cable':>9}{'implied':>11}{'range':>8}")
    print(f"{'':>11}{'':>6}{'(ticks)':>10}{'(mm)':>9}{'(ticks)':>8}{'(mm)':>9}"
          f"{'(ticks/s)':>11}{'':>8}")
    print("-" * 72)
    for r in rows:
        if not r.get("n"):
            print(f"{r['name']:>11}{'0':>6}   (no commands recorded)")
            continue
        flag = "OUT" if r["out_of_range"] else "ok"
        print(f"{r['name']:>11}{r['n']:>6}{r['max_step']:>10}"
              f"{r['max_step']*TICKS_TO_MM:>9.1f}{r['max_travel']:>8}"
              f"{r['max_travel']*TICKS_TO_MM:>9.1f}{r['implied']:>11.0f}{flag:>8}")
    bad = [r for r in rows if r.get("out_of_range")]
    if bad:
        print("\ntargets outside 0..4095:")
        for r in bad:
            over = (r["worst"] - (PERIOD - 1)) if r["worst"] > 0 else -r["worst"]
            print(f"   {r['name']}: {r['out_of_range']} commands, worst {r['worst']} "
                  f"(over by {over} ticks = {over * TICKS_TO_MM:.1f} mm)")
        print("   -> set_position will REFUSE these; the run would stop here")
    fast = [r for r in rows if r.get("implied", 0) > 7600]
    if fast:
        print("\nprimitives asking for more than the ~7600 ticks/s the servo can do:")
        for r in fast:
            print(f"   {r['name']}: {r['implied']:.0f} ticks/s implied "
                  f"({r['implied']*TICKS_TO_MM:.0f} mm/s) — the move will still be "
                  f"in flight when the next command lands")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="shoggoth_mini/configs/default_hardware.yaml")
    ap.add_argument("--only", default="", help="comma-separated subset of primitives")
    ap.add_argument("--calib-offset", type=int, default=0, metavar="TICKS",
                    help="shift the calibrated zero of every motor by this much, "
                         "to check that a retensioning offset does not push any "
                         "primitive out of range")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="noise_scale passed to the primitives (0 = deterministic)")
    ap.add_argument("--settle", type=float, default=1.0,
                    help="seconds to rest between primitives")
    ap.add_argument("--hz", type=float, default=50.0, help="sampler rate")
    ap.add_argument("--max-drift", type=int, default=60,
                    help="abort if a motor is this far from home after returning")
    ap.add_argument("--dry-run", action="store_true", help="phase 1 only")
    ap.add_argument("--csv", default="diagnostics/char_primitives.csv")
    ap.add_argument("--plot", default="diagnostics/char_primitives.png")
    ap.add_argument("--cmd-csv", default="diagnostics/char_primitives_cmd.csv")
    ap.add_argument("--static-csv", default="diagnostics/char_primitives_static.csv")
    ap.add_argument("--static-plot", default="diagnostics/char_primitives_static.png")
    args = ap.parse_args()

    names = [n.strip() for n in args.only.split(",") if n.strip()] or ORDER
    unknown = [n for n in names if n not in BEHAVIOUR]
    if unknown:
        raise SystemExit(f"unknown primitive(s) {unknown}; valid: {ORDER}")
    names = [n for n in ORDER if n in names]

    cfg = get_hardware_config(args.config)
    from shoggoth_mini.hardware.calibration import load_calibration
    calib = load_calibration(Path(cfg.calibration_file), MOTOR_NAMES)
    print(f"calibration on file: {calib}")

    if args.calib_offset:
        # Retensioning moves the zero. Everything downstream is an offset FROM the
        # zero, so the headroom to 0 and 4095 is not symmetric once the zero moves,
        # and the primitive with the largest travel loses margin first.
        calib = {m: v + args.calib_offset for m, v in calib.items()}
        print(f"applying --calib-offset {args.calib_offset:+d} -> {calib}")
        bad = {m: v for m, v in calib.items() if not (0 <= v < PERIOD)}
        if bad:
            print(f"\nREFUSING: offset puts the zero outside 0..{PERIOD-1}: {bad}")
            return 1
        head_up = min(PERIOD - 1 - v for v in calib.values())
        head_dn = min(v for v in calib.values())
        print(f"headroom from the shifted zero: {head_up} ticks up "
              f"({head_up * TICKS_TO_MM:.1f} mm), {head_dn} ticks down "
              f"({head_dn * TICKS_TO_MM:.1f} mm)")

    print("\n=== PHASE 1 — static, nothing moves ===")
    rows = static_pass(names, calib, args.noise, args.static_csv)
    print_static(rows)
    plot_static(args.static_csv, args.static_plot, rows, calib)

    if args.dry_run:
        print("\ndry run — phase 1 only, nothing moved.")
        return 0

    print("\n*** ALL TENDONS MUST BE DETACHED ***")
    print("Not merely untied. Phase 2 pays cable out of two motors at once, fast.")
    if input("Type 'detached' to run phase 2: ").strip().lower() != "detached":
        print("aborted — nothing moved.")
        return 1

    mc = MotorController(cfg)
    mc.connect()
    rec = Recorder(args.csv, FIELDS)
    crec = Recorder(args.cmd_csv, CMD_FIELDS)

    def on_command(t_abs, cmd):
        # Share the sampler's time origin so the two series align exactly.
        for m, v in cmd.items():
            crec.log(t=f"{t_abs - rec.t0:.4f}", phase=state["phase"],
                     motor=m, target=v)

    tee = TeeController(mc, on_command)

    # No external lock: MotorController serialises the bus with its own RLock.
    # An outer lock here held for the duration of each primitive starved the
    # sampler, so on 2026-08-18 not one primitive's motion was recorded — only
    # the gaps between them.
    state = {"phase": "init", "stop": False, "fault": None}

    def sampler():
        period = 1.0 / args.hz
        while not state["stop"]:
            try:
                pos = mc.get_positions()
                loads = {}
                for m in MOTOR_NAMES:
                    try:
                        # Present_Load has no MotorController accessor, so this
                        # touches the bus directly — and must therefore take the
                        # controller's OWN lock. Without it (2026-08-18) this read
                        # raced the main thread's writes and produced
                        # "[TxRxResult] Port is in use!" on the first homing read.
                        # Held per-read only, so the sampler is never starved the
                        # way an outer lock around a whole primitive starved it.
                        with mc._bus_lock:
                            v = mc._motor_bus.read("Present_Load", m)
                        loads[m] = int(v.item() if hasattr(v, "item") else v[0])
                    except Exception:
                        loads[m] = ""
                for m, p in pos.items():
                    rec.log(phase=state["phase"], motor=m, present=p,
                            load=loads.get(m, ""))
                    if not (0 <= p < PERIOD) and state["fault"] is None:
                        state["fault"] = f"motor {m} at {p}, outside 0..{PERIOD-1}"
            except Exception:
                pass
            time.sleep(period)

    th = threading.Thread(target=sampler, daemon=True)
    results = []

    def go_home():
        """Ramp home, never writing a target outside 0..4095.

        The first version computed `current + clip(home - current)` straight from
        the reading. On 2026-08-18 grab drove motor 2 to 4915, past the encoder
        fold; the reading then came back negative, this arithmetic produced a
        negative target, and a raw negative Goal_Position is decoded
        sign-magnitude — so -3217 became a target of -29551 and the motor ran away
        at full speed until it jammed. Fold the reading back into range first, and
        refuse to write anything out of range.
        """
        def read_folded():
            raw = mc.get_positions()
            return {m: int(raw[m]) % PERIOD for m in MOTOR_NAMES}, raw

        cur, raw = read_folded()
        if any(not (0 <= raw[m] < PERIOD) for m in MOTOR_NAMES):
            print(f"      note: reading outside 0..{PERIOD-1} {raw}, "
                  f"folded to {cur} before homing")
        # Tolerance must exceed the servo deadband. It was 3 ticks, but commands
        # under ~4 ticks produce NO motion and 4-8 ticks work about 40% of the
        # time (measured 2026-08-18), so the loop could never converge and spun
        # its full 400 iterations at 0.05 s = 20 s before silently giving up.
        # 12 ticks is 0.3 mm — far below the drift threshold this test cares
        # about, so nothing is lost by relaxing it.
        HOME_TOL = 12
        stalled = 0
        for _ in range(400):
            if max(abs(calib[m] - cur[m]) for m in MOTOR_NAMES) <= HOME_TOL:
                break
            nxt = {}
            for m in MOTOR_NAMES:
                t = cur[m] + int(np.clip(calib[m] - cur[m], -60, 60))
                if not (0 <= t < PERIOD):
                    raise RuntimeError(
                        f"homing computed an out-of-range target {t} for motor {m} "
                        f"from reading {cur[m]} — refusing to write it")
                nxt[m] = t
            tee.set_positions(nxt)
            time.sleep(0.05)
            prev = cur
            cur, raw = read_folded()
            # Give up loudly rather than grinding to the iteration cap.
            stalled = stalled + 1 if all(
                abs(cur[m] - prev[m]) < 2 for m in MOTOR_NAMES) else 0
            if stalled >= 10:
                print(f"      homing stalled {cur} vs {calib}, giving up")
                break
        else:
            print(f"      homing hit the iteration cap at {cur}")
        return cur

    try:
        print("\n=== PHASE 2 — live ===")
        state["phase"] = "home"
        th.start()
        start_home = go_home()
        print(f"home: {start_home}")

        for name in names:
            before = mc.get_positions()
            state["phase"] = name
            t0 = time.time()
            run_one(tee, name, calib, args.noise)
            dur = time.time() - t0

            state["phase"] = f"{name}/return"
            time.sleep(args.settle)
            after = go_home()
            drift = {m: after[m] - before[m] for m in MOTOR_NAMES}
            worst = max(abs(v) for v in drift.values())

            results.append(dict(name=name, duration=dur, drift=drift, worst=worst))
            flag = "   <-- DRIFT" if worst > args.max_drift else ""
            print(f"  {name:>11}  {dur:>5.2f}s   drift " +
                  " ".join(f"{m}:{drift[m]:+5d}" for m in MOTOR_NAMES) + flag)

            if state["fault"]:
                print(f"\n!! ABORT: {state['fault']}")
                break
            if worst > args.max_drift:
                print(f"\n!! ABORT: net drift {worst} ticks "
                      f"({worst*TICKS_TO_MM:.1f} mm) exceeds --max-drift")
                break

        state["phase"] = "final"
        go_home()

    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        state["stop"] = True
        th.join(timeout=2.0)
        rec.close()
        crec.close()
        mc.disconnect()

    if results:
        print(f"\n=== PHASE 2 RESULT ===")
        print(f"{'primitive':>11}{'dur':>7}{'peak vel':>11}{'mm/s':>8}"
              f"{'net drift':>11}{'mm':>8}")
        print("-" * 56)
        rows2 = load(args.csv)
        for r in results:
            sel = [x for x in rows2 if x["phase"] == r["name"]]
            pv = 0.0
            for m in MOTOR_NAMES:
                s = [x for x in sel if x["motor"] == m]
                if len(s) < 3:
                    continue
                t, p = col(s, "t"), col(s, "present")
                ok = ~np.isnan(p)
                if ok.sum() < 3:
                    continue
                dt = np.diff(t[ok])
                dt[dt <= 0] = np.nan
                v = np.abs(np.diff(p[ok]) / dt)
                if len(v):
                    pv = max(pv, float(np.nanmax(v)))
            print(f"{r['name']:>11}{r['duration']:>7.2f}{pv:>11.0f}"
                  f"{pv*TICKS_TO_MM:>8.0f}{r['worst']:>11}"
                  f"{r['worst']*TICKS_TO_MM:>8.2f}")

        drifted = [r for r in results if r["worst"] > 20]
        print("\ninterpretation:")
        if not drifted:
            print("    * every primitive returned home within 20 ticks (0.5 mm). "
                  "No wind or unwind events.")
        else:
            for r in drifted:
                print(f"    * {r['name']}: net drift {r['worst']} ticks "
                      f"({r['worst']*TICKS_TO_MM:.2f} mm) — the motor did not end "
                      f"where the code believes it did")
        if state["fault"]:
            print(f"    * FAULT: {state['fault']}")

        _plot(args.csv, args.plot, names, calib, args.cmd_csv)
        print(f"csv  -> {args.csv}")
    return 0


def _plot(csv_path, out_path, names, calib, cmd_csv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load(csv_path)
    fig, ax = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Primitive sweep — every designed motion, tendons detached",
                 fontsize=13)

    colours = {"1": "tab:blue", "2": "tab:orange", "3": "tab:green"}
    for m in MOTOR_NAMES:
        sel = [r for r in rows if r["motor"] == m]
        if not sel:
            continue
        t, p = col(sel, "t"), col(sel, "present")
        ax[0].plot(t, p, lw=1.0, color=colours[m], label=f"motor {m}")
        ax[1].plot(t, p - calib[m], lw=1.0, color=colours[m], label=f"motor {m}")

    # Shade each primitive so a drift is attributable to one of them.
    spans, last, t0 = [], None, None
    for r in rows:
        ph = r["phase"]
        if ph != last:
            if last is not None and t0 is not None:
                spans.append((last, t0, float(r["t"])))
            last, t0 = ph, float(r["t"])
    for i, (ph, a, b) in enumerate(spans):
        if ph in names:
            for axis in ax:
                axis.axvspan(a, b, color="gray", alpha=.12)
            ax[0].text((a + b) / 2, ax[0].get_ylim()[1], ph, fontsize=8,
                       ha="center", va="top", rotation=90)

    ax[0].axhline(0, color="tab:red", lw=1, ls="--", alpha=.6)
    ax[0].axhline(PERIOD - 1, color="tab:red", lw=1, ls="--", alpha=.6,
                  label="0 / 4095")
    ax[0].set_ylabel("Present_Position (ticks)")
    ax[0].set_title("absolute position — shaded bands are primitives", fontsize=10)
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].grid(alpha=.3)

    ax[1].axhline(0, color="gray", lw=1)
    ax[1].set_ylabel("offset from calibrated zero (ticks)")
    ax[1].set_title("offset from home — every primitive should return to zero; "
                    "a step that does not is a wind or unwind event", fontsize=10)
    ax[1].grid(alpha=.3)
    sec = ax[1].secondary_yaxis("right", functions=(lambda t: t * TICKS_TO_MM,
                                                    lambda s: s / TICKS_TO_MM))
    sec.set_ylabel("cable (mm)")

    # panel 3: tracking error. The commanded value is a zero-order hold — the
    # servo holds the last target until a new one arrives — so it is stepped onto
    # the sampler's timestamps with np.searchsorted rather than interpolated.
    cmd_rows = load(cmd_csv) if cmd_csv and Path(cmd_csv).exists() else []
    if cmd_rows:
        summary = {}
        for m in MOTOR_NAMES:
            csel = [r for r in cmd_rows if r["motor"] == m]
            psel = [r for r in rows if r["motor"] == m]
            if len(csel) < 2 or len(psel) < 2:
                continue
            ct, cv = col(csel, "t"), col(csel, "target")
            pt, pv = col(psel, "t"), col(psel, "present")
            order = np.argsort(ct)
            ct, cv = ct[order], cv[order]
            idx = np.searchsorted(ct, pt, side="right") - 1
            valid = idx >= 0
            err = np.full(len(pt), np.nan)
            err[valid] = pv[valid] - cv[idx[valid]]
            ax[2].plot(pt, err, lw=1.0, color=colours[m], label=f"motor {m}")
            fin = err[~np.isnan(err)]
            if len(fin):
                summary[m] = (float(np.sqrt(np.mean(fin ** 2))),
                              float(np.max(np.abs(fin))))
        ax[2].axhline(0, color="gray", lw=1)
        for band, c in ((36, "tab:orange"), (100, "tab:red")):
            for sgn in (1, -1):
                ax[2].axhline(sgn * band, color=c, lw=0.9, ls="--", alpha=.6)
        ax[2].plot([], [], color="tab:orange", ls="--", label="±36 (measured overshoot)")
        ax[2].plot([], [], color="tab:red", ls="--", label="±100 (position_tolerance)")
        title = "tracking error, achieved - commanded"
        if summary:
            title += "   |   " + "  ".join(
                f"m{m}: RMS {v[0]:.0f}, peak {v[1]:.0f}" for m, v in sorted(summary.items()))
        ax[2].set_title(title + "\nsustained error means the primitive is asking "
                        "for more than the servo can deliver", fontsize=10)
        ax[2].legend(fontsize=8, loc="upper right", ncol=2)
    else:
        ax[2].text(0.5, 0.5, "no command log", ha="center", va="center",
                   transform=ax[2].transAxes, color="gray")
        ax[2].set_title("tracking error", fontsize=10)
    ax[2].set_ylabel("error (ticks)")
    ax[2].set_xlabel("time (s)")
    ax[2].grid(alpha=.3)
    sec3 = ax[2].secondary_yaxis("right", functions=(lambda t: t * TICKS_TO_MM,
                                                     lambda s: s / TICKS_TO_MM))
    sec3.set_ylabel("cable (mm)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    print(f"plot -> {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
