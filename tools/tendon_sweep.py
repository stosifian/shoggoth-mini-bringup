"""Deterministic, symmetric workspace sweep — a repeatable mechanical test.

`idle` is randomised by design (amplitude, period and direction are all re-rolled
per run), so it cannot answer "is the robot asymmetric?" — you cannot separate a
mechanical bias from the RNG. This drives a fixed star of cursor directions at a
fixed magnitude, returning to the calibrated rest pose between every spoke, and
reports the tick error on each return.

What the numbers mean:
  * return error near zero everywhere  -> nothing slipping; visible bias is geometry
  * error that grows run over run      -> something is creeping (knot, horn, spool)
  * error biased to one spoke/motor    -> that tendon is hitting a limit or slipping

Motion is rate-limited and always relative to the calibration file, same as the
control stack. Nothing is written to the calibration file.

  python tools/tendon_sweep.py                    # 6 spokes at |c|=0.18
  python tools/tendon_sweep.py --magnitude 0.12
  python tools/tendon_sweep.py --spokes 12 --cycles 3
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.configs import get_hardware_config  # noqa: E402
from shoggoth_mini.hardware.motors import MotorController  # noqa: E402
from shoggoth_mini.control.geometry import cursor_to_motor_positions  # noqa: E402
from shoggoth_mini.common.constants import MOTOR_NAMES  # noqa: E402

TICKS_TO_MM = 0.11 / 4096 * 1000


def glide(mc, calib, target_cursor, step_ticks, hz=100.0):
    """Move to a cursor position in rate-limited steps from wherever we are."""
    target, _ = cursor_to_motor_positions(
        cursor_pos=np.array(target_cursor, dtype=float), calibrated_ticks_map=calib
    )
    target = {m: int(target[m]) for m in MOTOR_NAMES}
    current = {m: mc.get_position(m) for m in MOTOR_NAMES}

    while True:
        deltas = {m: target[m] - current[m] for m in MOTOR_NAMES}
        if all(abs(d) <= step_ticks for d in deltas.values()):
            mc.set_positions(target)
            break
        for m in MOTOR_NAMES:
            d = deltas[m]
            current[m] += max(-step_ticks, min(step_ticks, d))
        mc.set_positions(current)
        time.sleep(1.0 / hz)
    time.sleep(0.35)  # settle before measuring


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="shoggoth_mini/configs/default_hardware.yaml")
    ap.add_argument("--magnitude", type=float, default=0.18,
                    help="cursor magnitude per spoke (measured useful max ~0.25)")
    ap.add_argument("--spokes", type=int, default=6)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--step", type=int, default=15, help="max ticks per loop")
    ap.add_argument("--arc", action="store_true",
                    help="sweep a limited arc back and forth instead of a full circle, "
                         "keeping the tip in front of the camera (a 360 sweep takes it "
                         "behind the dome and out of the stereo overlap). Pair with "
                         "debug-perception running in another terminal.")
    ap.add_argument("--arc-center", type=float, default=330.0, metavar="DEG",
                    help="arc centre. Default 330 = motor 2's pull axis, which moves "
                         "the tip toward the camera on this build.")
    ap.add_argument("--arc-span", type=float, default=120.0, metavar="DEG",
                    help="total arc width (default 120 = centre +/-60)")
    ap.add_argument("--arc-steps", type=int, default=60,
                    help="angular resolution of one traverse")
    ap.add_argument("--arc-dwell", type=float, default=0.15,
                    help="seconds to pause at each step (slower = easier to watch)")
    ap.add_argument("--hold", type=float, default=None, metavar="DEG",
                    help="drive to this angle and HOLD until Enter, for inspecting "
                         "marker lines under load, then return to rest")
    args = ap.parse_args()

    if args.magnitude > 0.25:
        print(f"refusing |c|={args.magnitude}: above the measured useful range (~0.25)")
        return 1

    cfg = get_hardware_config(args.config)
    mc = MotorController(cfg)
    mc.connect()
    calib = mc.get_calibration_data()
    if not calib or all(v == 0 for v in calib.values()):
        print("no calibration loaded — run `calibrate` first")
        return 1

    # PRE-FLIGHT against the limits that ACTUALLY apply:
    #   * max_travel_ticks — how far from the calibrated zero any motion may go.
    #     This is the real constraint: the failure mode is commanding more cable
    #     travel than the tendon can absorb.
    #   * the servo's own Min/Max_Angle_Limit, when it reports bounded.
    # An earlier version checked 0..4095. That was wrong: these servos are
    # multi-turn and honour targets past 4095 (tested 2026-08-13), so it refused
    # perfectly legal sweeps.
    #   * every target inside 0..4095. The servo does track multi-turn, so this is
    #     a POLICY bound rather than a hardware one — but it is the bound that was
    #     agreed after testing (2026-08-18), and it is the only check here that
    #     works when the other two are unset. Both of them were inert on
    #     2026-08-18 (max_travel_ticks absent, _position_limits removed), so a
    #     calibration of -1548 on motor 2 sailed through and the sweep commanded
    #     a 4234-tick move with the tendons threaded.
    travel_limit = int(getattr(cfg, "max_travel_ticks", 0) or 0)
    servo_limits = {m: lim for m, lim in (getattr(mc, "_position_limits", {}) or {}).items()
                    if lim is not None}

    bad_zero = {m: z for m, z in calib.items() if not (0 <= z < 4096)}
    if bad_zero:
        print("\nREFUSING TO RUN — calibrated zero outside the encoder range:")
        for m, z in sorted(bad_zero.items()):
            print(f"   motor {m}: zero is {z}, must be 0..4095")
        print("\nEvery target is an offset from these values, so all of them are "
              "unreachable positions. Re-run calibration.")
        mc.disconnect()
        return 1

    problems = {}
    for i in range(max(args.spokes, 12)):
        ang = 2 * np.pi * i / max(args.spokes, 12)
        cur = (args.magnitude * np.cos(ang), args.magnitude * np.sin(ang))
        tgt, _ = cursor_to_motor_positions(
            cursor_pos=np.array(cur, dtype=float), calibrated_ticks_map=calib
        )
        for m in MOTOR_NAMES:
            t = int(tgt[m])
            if not (0 <= t < 4096):
                over = max(t - 4095, -t, 0)
                prev = problems.get(m, {}).get("range", (0, t))
                if over > prev[0]:
                    problems.setdefault(m, {})["range"] = (over, t)
            if travel_limit > 0:
                over = abs(t - calib[m]) - travel_limit
                if over > 0:
                    problems.setdefault(m, {})["travel"] = max(
                        problems.get(m, {}).get("travel", 0), over)
            if m in servo_limits:
                lo, hi = servo_limits[m]
                over = max(t - hi, lo - t, 0)
                if over > 0:
                    problems.setdefault(m, {})["servo"] = max(
                        problems.get(m, {}).get("servo", 0), over)
    if problems:
        print(f"\nREFUSING TO RUN at |c|={args.magnitude} — commands would be clamped:")
        for m, kinds in sorted(problems.items()):
            if "range" in kinds:
                over, worst = kinds["range"]
                print(f"   motor {m}: target {worst} is outside 0..4095 by "
                      f"{over} ticks ({over * TICKS_TO_MM:.1f} mm of cable)")
            if "travel" in kinds:
                print(f"   motor {m}: exceeds the +/-{travel_limit} tick travel limit "
                      f"by {kinds['travel']} ticks "
                      f"({kinds['travel'] * TICKS_TO_MM:.1f} mm of cable)")
            if "servo" in kinds:
                print(f"   motor {m}: exceeds the servo's own limits by "
                      f"{kinds['servo']} ticks")
        print("\nReduce --magnitude, or raise max_travel_ticks in the hardware config")
        print("if the tendons genuinely have that much travel.")
        mc.disconnect()
        return 1

    print(f"\ncalibration: {calib}")

    if args.arc:
        lo = args.arc_center - args.arc_span / 2.0
        hi = args.arc_center + args.arc_span / 2.0
        print(f"arc sweep {lo:.0f} -> {hi:.0f} deg at |c|={args.magnitude}, "
              f"{args.cycles} cycle(s)")
        print("Tip stays in front of the camera throughout — watch the Tip counter in "
              "debug-perception.\n")
        try:
            for cycle in range(args.cycles):
                seq = list(np.linspace(lo, hi, args.arc_steps))
                seq += seq[::-1][1:]  # and back, without repeating the endpoint
                for deg in seq:
                    a = np.radians(deg)
                    cursor = (args.magnitude * np.cos(a), args.magnitude * np.sin(a))
                    glide(mc, calib, cursor, args.step)
                    time.sleep(args.arc_dwell)
                    off = {m: mc.get_position(m) - calib[m] for m in MOTOR_NAMES}
                    print(f"  cycle {cycle+1}/{args.cycles}  angle {deg:6.1f}deg  "
                          f"offsets {off}    ", end="\r", flush=True)
        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            glide(mc, calib, (0.0, 0.0), args.step)
            err = {m: mc.get_position(m) - calib[m] for m in MOTOR_NAMES}
            print(f"\n\nreturned to rest, error: {err}  "
                  f"({max(abs(v) for v in err.values())*TICKS_TO_MM:.2f} mm)")
            mc.disconnect()
        return 0

    if args.hold is not None:
        angle = np.radians(args.hold)
        cursor = (args.magnitude * np.cos(angle), args.magnitude * np.sin(angle))
        print(f"holding {args.hold:.0f}deg at |c|={args.magnitude} "
              f"(cursor {cursor[0]:+.3f}, {cursor[1]:+.3f})")
        try:
            glide(mc, calib, cursor, args.step)
            held = {m: mc.get_position(m) - calib[m] for m in MOTOR_NAMES}
            print(f"tick offsets under load: {held}")
            print("\nInspect the shaft->horn->roller marker lines now.")
            print("Press Enter to return to rest...")
            input()
        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            glide(mc, calib, (0.0, 0.0), args.step)
            err = {m: mc.get_position(m) - calib[m] for m in MOTOR_NAMES}
            peak = max(abs(v) for v in err.values())
            print(f"return error: {err}  ({peak * TICKS_TO_MM:.2f} mm)")
            print("A large return error after holding = something moved under load.")
            mc.disconnect()
        return 0

    print(f"{args.spokes} spokes at |c|={args.magnitude}, {args.cycles} cycle(s)\n")
    print(f"{'spoke':>7} {'angle':>7}   {'return error (ticks)':>28}   {'max mm':>7}")
    print("-" * 60)

    worst = 0
    try:
        for cycle in range(args.cycles):
            for i in range(args.spokes):
                angle = 2 * np.pi * i / args.spokes
                cursor = (args.magnitude * np.cos(angle), args.magnitude * np.sin(angle))

                glide(mc, calib, cursor, args.step)
                glide(mc, calib, (0.0, 0.0), args.step)

                err = {m: mc.get_position(m) - calib[m] for m in MOTOR_NAMES}
                peak = max(abs(v) for v in err.values())
                worst = max(worst, peak)
                flag = "  <-- CHECK" if peak > 30 else ""
                print(f"{i + 1:>7} {np.degrees(angle):>6.0f}deg   "
                      f"{str(err):>28}   {peak * TICKS_TO_MM:>6.2f}{flag}")
            if args.cycles > 1:
                print(f"  -- end cycle {cycle + 1} --")
    except KeyboardInterrupt:
        print("\ninterrupted — returning to rest")
    finally:
        glide(mc, calib, (0.0, 0.0), args.step)
        final = {m: mc.get_position(m) - calib[m] for m in MOTOR_NAMES}
        print(f"\nfinal offset from calibration: {final}  "
              f"({max(abs(v) for v in final.values()) * TICKS_TO_MM:.2f} mm)")
        print(f"worst return error during sweep: {worst} ticks "
              f"({worst * TICKS_TO_MM:.2f} mm)")
        mc.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
