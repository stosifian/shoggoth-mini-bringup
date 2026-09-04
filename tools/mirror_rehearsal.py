"""Rehearse orchestrate_mirror's motion, TENDONS ATTACHED, one state at a time.

char_primitive_sweep answers "is any designed motion dangerous" and demands the
tendons be detached to answer it. This answers a narrower question -- "does the
mirror machine's motion behave with the tendons on" -- and is built to be run
with them attached, which means it earns that by being conservative rather than
by being brave.

It differs from the sweep in three ways that matter:

  * It goes through MotionWorker, posting the same BodyAction objects
    orchestrate_mirror posts, so the hold-and-release of grab, the looping of
    sustained bodies and the one-shot gestures are exercised as they actually
    run -- not as a list of primitives called in sequence.
  * The bodies come from the States table, so if you remap a state this follows.
  * States are ordered gentlest first by measured command rate, and it stops at
    the first sign of trouble rather than completing the set.

    python tools/mirror_rehearsal.py --dry-run          # nothing moves
    python tools/mirror_rehearsal.py --upto NEUTRAL     # stop after slow_circle
    python tools/mirror_rehearsal.py --pause            # confirm each state

PHASE 1 is static and needs no hardware. It drives the real primitives through a
recording stand-in and reports, per state, every commanded position, the largest
single step, and the implied speed from the primitive's own sleeps. Anything
outside 0..4095 or above the servo ceiling is reported before a motor is touched.

PHASE 2 executes for real while a background thread samples all three motors. Per
state it reports NET DRIFT -- position after returning home versus before. The
code expects zero. A slipped roller or a lost turn shows up here and nowhere
else, and with the tendons ATTACHED that is the number that matters most,
because a tendon under load is what makes a lost step mechanical rather than
merely numerical.

SAFETY
  * --dry-run moves nothing.
  * aborts on any commanded position outside 0..4095, on net drift over
    --max-drift, and on any exception from a primitive.
  * returns home between states, so drift is judged per state rather than
    accumulating unnoticed.
  * grab depth is derived from the live calibration and floored; see
    primitives.max_grab_magnitude.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from char_primitive_sweep import RecordingController  # noqa: E402
from shoggoth_mini.affect.fsm import load_tables  # noqa: E402
from shoggoth_mini.common.constants import MOTOR_NAMES  # noqa: E402
from shoggoth_mini.orchestrator.mirror import (  # noqa: E402
    LOOP_GAP_S, BodyAction, MotionWorker, body_for)

TICKS_TO_MM = 26.9 / 1000.0     # cable mm per tick, as char_primitive_sweep uses
SERVO_CEILING = 7600            # measured ticks/s
# A primitive cannot be interrupted, so stopping means waiting it out. The wave
# packets run about 45 s each; anything shorter reports a half-played motion.
STOP_TIMEOUT_S = 180.0

# Gentlest first, by the implied command rate char_primitive_sweep measures.
# SAD carries the grab, so it also exercises the release on the way out; ANGRY
# shares that body and is skipped rather than repeating the deepest motion.
ORDER = ["ALONE", "NOTICING", "NEUTRAL", "CONTENT", "EXCITED", "SAD", "YES", "NO"]


def rehearse(worker, state, body, loops, settle, cap=None):
    """Post a body the way the orchestrator does, and let it run.

    `cap` bounds the time spent, because a primitive cannot be interrupted and
    two of them run the better part of a minute per play. Capping means a state
    may be cut mid-motion, which is fine for a rehearsal and NOT how the
    orchestrator behaves.
    """
    worker.post(body)
    want = loops * (settle + LOOP_GAP_S) if body.loop else settle
    time.sleep(min(want, cap) if cap else want)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", type=Path, default=Path("shoggoth-mirror-state"))
    ap.add_argument("--dry-run", action="store_true",
                    help="phase 1 only; nothing moves")
    ap.add_argument("--upto", default=None,
                    help="stop after this state (see ORDER)")
    ap.add_argument("--only", default=None, help="comma-separated states")
    ap.add_argument("--pause", action="store_true",
                    help="wait for Enter before each state")
    ap.add_argument("--loops", type=int, default=3,
                    help="repeats to allow a looping body")
    ap.add_argument("--max-seconds", type=float, default=20.0,
                    help="cap per state. The long ones are long because the "
                         "PRIMITIVE is: a wave packet runs ~45 s per play, so "
                         "the loop count is not what makes them slow")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="seconds allowed per primitive")
    ap.add_argument("--max-drift", type=int, default=60,
                    help="abort if a state's net drift exceeds this, in ticks")
    args = ap.parse_args()

    _, _, states, states_rows, _, _ = load_tables(args.tables)
    order = [s for s in ORDER if s in states]
    if args.only:
        want = {x.strip().upper() for x in args.only.split(",")}
        order = [s for s in order if s in want]
    elif args.upto:
        if args.upto.upper() not in order:
            sys.exit(f"--upto {args.upto} not in {order}")
        order = order[:order.index(args.upto.upper()) + 1]

    bodies = {s: body_for(s, states_rows) for s in order}
    print(f"  tables: {args.tables}   states: {', '.join(order)}")
    for s in order:
        b = bodies[s]
        kind = "HOLD+release" if b.on_exit else ("loop" if b.loop else "one-shot")
        print(f"    {s:9} {str(b.primitive):16} {kind}")

    # ---------------- phase 1: static ---------------------------------------
    from shoggoth_mini.configs import get_hardware_config
    hw_yaml = HERE.parent / "shoggoth_mini" / "configs" / "default_hardware.yaml"
    hw = get_hardware_config(str(hw_yaml) if hw_yaml.exists() else None)

    from shoggoth_mini.hardware.calibration import load_calibration
    calib = load_calibration(hw.calibration_file, MOTOR_NAMES)

    print(f"\n=== PHASE 1 — static, nothing moves ===\n")
    print(f"  calibration {calib}")
    print(f"\n  {'state':10}{'cmds':>6}{'max step':>10}{'lowest':>8}{'highest':>9}"
          f"{'margin':>8}{'ticks/s':>10}{'secs':>7}  verdict")
    worst, timing = None, {}
    for s in order:
        rec = RecordingController(calib)
        w = MotionWorker(rec, dry_run=False)
        w.start()
        b = bodies[s]
        t_start = time.time()
        rehearse(w, s, b, args.loops, 0.05, args.max_seconds)
        if b.on_exit:
            w.post(BodyAction("_after", None))     # forces the release
            time.sleep(0.3)
        # generous: a primitive cannot be cancelled, and a short wait would
        # report a half-played motion as if it were the whole thing
        w.stop(timeout=STOP_TIMEOUT_S)
        timing[s] = time.time() - t_start

        cmds = rec.commands
        if not cmds:
            print(f"  {s:10}{0:>6}   (no commands)")
            continue
        vals = [v for _, c in cmds for v in c.values()]
        lo, hi = min(vals), max(vals)
        step = rate = 0
        for (t0, a), (t1, bb) in zip(cmds, cmds[1:]):
            d = max(abs(bb[m] - a.get(m, bb[m])) for m in bb)
            step = max(step, d)
            dt = max(t1 - t0, 1e-4)
            rate = max(rate, d / dt)
        margin = min(lo, 4095 - hi)
        bad = margin < 0 or rate > SERVO_CEILING
        verdict = "ok" if not bad else ("RANGE" if margin < 0 else "OVER CEILING")
        if bad:
            worst = worst or s
        print(f"  {s:10}{len(cmds):>6}{step:>10}{lo:>8}{hi:>9}{margin:>8}"
              f"{rate:>10.0f}{timing[s]:>7.1f}  {verdict}")

    if worst:
        print(f"\n!! {worst} failed the static pass. Not proceeding.")
        return 1
    total = sum(timing.values())
    print("\n  static pass clean: every command inside 0..4095, all under the "
          f"{SERVO_CEILING} ticks/s ceiling")
    print(f"  phase 2 will take about {total/60:.1f} min of motion "
          f"(+ home and settle between states)")

    if args.dry_run:
        print("\ndry run — phase 1 only, nothing moved.")
        return 0

    # ---------------- phase 2: for real -------------------------------------
    print(f"\n=== PHASE 2 — TENDONS ATTACHED, motors will move ===")
    print(f"  port {hw.port}   tick_sign {hw.tick_sign}")
    print(f"  net drift over {args.max_drift} ticks "
          f"({args.max_drift*TICKS_TO_MM:.1f} mm) aborts")
    if input("\n  type 'go' to move the robot: ").strip().lower() != "go":
        print("  aborted.")
        return 1

    from shoggoth_mini.hardware.motors import MotorController
    mc = MotorController(hw)
    mc.connect()
    stop = threading.Event()
    peak = {m: 0 for m in MOTOR_NAMES}

    def sampler():
        last = None
        while not stop.is_set():
            try:
                p = mc.get_positions()
            except Exception:
                time.sleep(0.02); continue
            if last:
                for m in MOTOR_NAMES:
                    peak[m] = max(peak[m], abs(p[m] - last[m]))
            last = p
            time.sleep(0.02)

    th = threading.Thread(target=sampler, daemon=True); th.start()
    results, aborted = [], None
    try:
        for s in order:
            if args.pause:
                input(f"\n  Enter to run {s} ({bodies[s].primitive}) ")
            mc.reset_to_calibrated_positions()
            time.sleep(0.6)
            before = mc.get_positions()

            w = MotionWorker(mc, dry_run=False)
            w.start()
            t0 = time.time()
            try:
                # phase 1 measured how long this body actually takes; wait that
                # long rather than a guess, so nothing is cut off mid-motion
                rehearse(w, s, bodies[s], args.loops,
                         max(args.settle, timing.get(s, args.settle)),
                         args.max_seconds)
            finally:
                w.stop(timeout=STOP_TIMEOUT_S)   # also releases a held grip
            dur = time.time() - t0

            mc.reset_to_calibrated_positions()
            time.sleep(0.6)
            after = mc.get_positions()
            drift = {m: after[m] - before[m] for m in MOTOR_NAMES}
            wd = max(abs(v) for v in drift.values())
            results.append((s, dur, drift, wd))
            flag = "   <-- DRIFT" if wd > args.max_drift else ""
            print(f"  {s:10} {dur:5.1f}s  drift " +
                  " ".join(f"{m}:{drift[m]:+5d}" for m in MOTOR_NAMES) + flag)
            if wd > args.max_drift:
                aborted = s
                print(f"\n!! ABORT after {s}: net drift {wd} ticks "
                      f"({wd*TICKS_TO_MM:.1f} mm) exceeds --max-drift")
                break
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        stop.set(); th.join(timeout=1.0)
        try:
            mc.reset_to_calibrated_positions()
        finally:
            mc.disconnect()
        print("  motors returned home and disconnected")

    print(f"\n  {'state':10}{'net drift':>11}{'mm':>8}")
    for s, dur, drift, wd in results:
        print(f"  {s:10}{wd:>11}{wd*TICKS_TO_MM:>8.1f}")
    print(f"  largest single sampled step per motor: " +
          " ".join(f"{m}:{peak[m]}" for m in MOTOR_NAMES))
    return 1 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
