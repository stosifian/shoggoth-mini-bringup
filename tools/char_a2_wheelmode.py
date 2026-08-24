"""TEST A2 — characterise Present_Position by spinning the spool under power.

Test A asked you to turn the spool by hand with torque off. That is impractical: the
STS3215 has a metal gearbox and is stiff to backdrive at the output even when limp
(confirmed 2026-08-17 — torque really was 0, the spool just would not turn).

So this drives the spool instead, in WHEEL mode via Goal_Speed. Crucially it issues
NO position commands, so the modulo-4096 shortest-path behaviour that has caused
every failure this week cannot occur here — there is no target to take a short arc
to. It only sets a speed and watches the position register.

WHAT IT ANSWERS (same questions as Test A)
  * does Present_Position stay within 0..4095, exceed it, or go negative?
  * where does it fold, and by how much?
  * how many ticks does one mechanical revolution actually report?
  * does the answer differ between directions?

SAFETY
  * TENDON MUST BE DETACHED (or fully free). Continuous rotation will otherwise
    spool wire in or out without limit — that is how rollers get stripped.
  * low default speed, fixed duration, stops on Ctrl+C
  * restores Mode and leaves Goal_Speed at 0 on exit
  * torque is enabled AFTER the Mode write, because writing Mode clears it
    (firmware interlock: writing Mode resets Torque_Enable to 0)

  python tools/char_a2_wheelmode.py --motor 2
  python tools/char_a2_wheelmode.py --motor 2 --speed 200 --seconds 15
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.configs import get_hardware_config  # noqa: E402
from shoggoth_mini.hardware.motors import MotorController  # noqa: E402
from shoggoth_mini.common.constants import MOTOR_NAMES  # noqa: E402
from char_common import (  # noqa: E402
    Recorder, render_position_trace, summarise_position, col, load, unwrap_ticks,
)

FIELDS = ["t", "motor", "phase", "present", "goal_speed", "speed", "load",
          "torque", "mode", "min_limit", "max_limit"]
WHEEL_MODE, SERVO_MODE = 1, 0


def signed_speed(direction: int, magnitude: int) -> int:
    """Goal_Speed is sign-magnitude in wheel mode: CW uses -(1024 - magnitude)."""
    magnitude = min(abs(magnitude), 1023)
    return magnitude if direction >= 0 else -(1024 - magnitude)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="shoggoth_mini/configs/default_hardware.yaml")
    ap.add_argument("--motor", default="2", choices=MOTOR_NAMES)
    ap.add_argument("--speed", type=int, default=250,
                    help="Goal_Speed magnitude (calibrate uses 800; 250 is slow)")
    ap.add_argument("--seconds", type=float, default=12.0, help="per direction")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--csv", default="diagnostics/char_a2.csv")
    ap.add_argument("--plot", default="diagnostics/char_a2.png")
    args = ap.parse_args()

    mc = MotorController(get_hardware_config(args.config))
    mc.connect()
    m = args.motor
    bus = mc._motor_bus

    def rd(field):
        try:
            v = bus.read(field, m)
            return int(v.item() if hasattr(v, "item") else v[0])
        except Exception:
            return None

    found = {k: rd(k) for k in ("Mode", "Min_Angle_Limit", "Max_Angle_Limit",
                                "Torque_Enable")}
    print(f"\nmotor {m} as found: {found}")
    print("\n*** THE TENDON ON THIS MOTOR MUST BE DETACHED OR FREE ***")
    print("This rotates continuously and would otherwise spool wire without limit.")
    if input("Type 'free' to continue: ").strip().lower() != "free":
        print("aborted — nothing moved.")
        mc.disconnect()
        return 1

    rec = Recorder(args.csv, FIELDS)
    period = 1.0 / args.hz

    def spin(direction, label):
        gs = signed_speed(direction, args.speed)
        # Mode first, then torque: the Mode write clears Torque_Enable.
        bus.write("Mode", WHEEL_MODE, m)
        bus.write("Goal_Speed", 0, m)
        bus.write("Torque_Enable", 1, m)
        bus.write("Goal_Speed", gs, m)
        t_end = time.time() + args.seconds
        last = 0.0
        while time.time() < t_end:
            p = rd("Present_Position")
            rec.log(motor=m, phase=label, present=p, goal_speed=gs,
                    speed=rd("Present_Speed"), load=rd("Present_Load"),
                    torque=rd("Torque_Enable"), mode=WHEEL_MODE,
                    min_limit=found["Min_Angle_Limit"],
                    max_limit=found["Max_Angle_Limit"])
            now = time.time()
            if now - last > 0.25:
                print(f"  {label}: present={p}  samples={rec.rows}    ",
                      end="\r", flush=True)
                last = now
            time.sleep(period)
        bus.write("Goal_Speed", 0, m)
        time.sleep(0.4)

    try:
        print(f"\nspinning CCW at Goal_Speed {args.speed} for {args.seconds:.0f}s ...")
        spin(+1, "ccw")
        print(f"\nspinning CW  for {args.seconds:.0f}s ...")
        spin(-1, "cw")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        try:
            bus.write("Goal_Speed", 0, m)
            bus.write("Mode", SERVO_MODE, m)
            # Deliberately NOT re-enabling torque or pinning Goal_Position: pinning
            # a read value back as a target is what caused an abrupt move on
            # 2026-08-17, and the semantics of that register pair are exactly what
            # this test exists to establish.
            print("\nGoal_Speed 0, Mode restored to servo. Torque left OFF.")
        except Exception as e:
            print(f"cleanup issue: {e}")
        rec.close()
        mc.disconnect()

    rows = load(args.csv)
    pos = col(rows, "present")
    stats = summarise_position(pos)
    print(f"\n=== TEST A2 RESULT — motor {m}, {len(rows)} samples ===")
    for k, v in stats.items():
        print(f"    {k:<18} {v}")
    unw = unwrap_ticks(pos)
    finite = unw[~__import__("numpy").isnan(unw)]
    if len(finite) > 1:
        span = finite.max() - finite.min()
        print(f"    {'unwrapped span':<18} {span:.0f} ticks = {span/4096:.2f} turns")

    print("\ninterpretation:")
    if stats.get("negative_samples", 0):
        print("    * reading GOES NEGATIVE -> not an unsigned 0..4095 field")
    if stats.get("above_period", 0):
        print("    * reading EXCEEDS 4095 -> accumulates past one turn")
    if stats.get("wraps", 0) and not stats.get("above_period", 0):
        print(f"    * folds at the turn boundary ({stats['wraps']} wraps), stays in range")
    if not any((stats.get("negative_samples"), stats.get("above_period"),
                stats.get("wraps"))):
        print("    * stayed inside one turn without wrapping — spin longer or faster")

    render_position_trace(args.csv, args.plot,
                          title=f"Test A2 (wheel mode) — motor {m}, "
                                f"Goal_Speed ±{args.speed}")
    print(f"csv  -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
