"""Park the spools at a known tick position — a clean starting point for re-threading.

Use when the wire has come off the rollers and you want a defined reference before
re-installing, rather than whatever arbitrary position the spools ended up in.

Default target is 2048: the middle of the 0..4095 single-turn encoder range, which
leaves maximum travel in BOTH directions for the pre-roll. Parking near 0 or 4095
means the encoder wraps partway through calibration.

SAFETY — written after a fast unramped move ate the cable on 2026-08-13:
  * moves are ramped in small increments, deliberately slow
  * it prints how far each motor will move, in mm of cable, BEFORE moving
  * it asks for confirmation
  * paying cable OUT with no tension is what unspools rollers. Keep light tension
    on the wire by hand, or take the wire off the spool first and park empty.

  python tools/park_motors.py                 # preview only
  python tools/park_motors.py --apply
  python tools/park_motors.py --target 2048 --motors 1,3 --apply
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.configs import get_hardware_config  # noqa: E402
from shoggoth_mini.hardware.motors import MotorController  # noqa: E402
from shoggoth_mini.common.constants import MOTOR_NAMES  # noqa: E402

TICKS_TO_MM = 0.11 / 4096 * 1000


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="shoggoth_mini/configs/default_hardware.yaml")
    ap.add_argument("--target", type=int, default=2048,
                    help="tick position to park at (default 2048 = encoder centre)")
    ap.add_argument("--motors", default="1,2,3")
    # 3 ticks at 50 Hz = 150 ticks/s = 4.0 mm of cable per second. Deliberately
    # slow: the move that seized the tentacle on 2026-08-13 ran at full servo speed,
    # and paying cable out faster than the wire can stay seated is what unspools the
    # rollers.
    ap.add_argument("--step", type=int, default=3,
                    help="ticks per increment (default 3; with --hz 50 = 150 ticks/s)")
    ap.add_argument("--hz", type=float, default=50.0,
                    help="increments per second")
    ap.add_argument("--apply", action="store_true", help="actually move")
    args = ap.parse_args()

    if not 0 <= args.target <= 4095:
        raise SystemExit(f"--target {args.target} outside the encoder range 0..4095")

    targets = [m.strip() for m in args.motors.split(",") if m.strip()]
    if any(m not in MOTOR_NAMES for m in targets):
        raise SystemExit(f"unknown motor in {targets}; valid: {MOTOR_NAMES}")

    mc = MotorController(get_hardware_config(args.config))
    mc.connect()
    current = {m: mc.get_position(m) for m in MOTOR_NAMES}

    print(f"\npark target: {args.target} ticks "
          f"({'encoder centre' if args.target == 2048 else 'custom'})")
    print(f"\n{'motor':>6} {'now':>7} {'target':>8} {'delta':>8} {'cable':>12}")
    print("-" * 46)
    worst = 0
    for m in MOTOR_NAMES:
        if m in targets:
            d = args.target - current[m]
            worst = max(worst, abs(d))
            direction = "wind IN" if d > 0 else ("pay OUT" if d < 0 else "")
            print(f"{m:>6} {current[m]:>7} {args.target:>8} {d:>+8} "
                  f"{d * TICKS_TO_MM:>+7.1f} mm {direction}")
        else:
            print(f"{m:>6} {current[m]:>7} {'—':>8} {'—':>8} {'untouched':>12}")

    secs = worst / max(args.step * args.hz, 1)
    rate = args.step * args.hz
    print(f"\nlargest move {worst} ticks ({worst * TICKS_TO_MM:.0f} mm of cable), "
          f"~{secs:.0f} s at {rate:.0f} ticks/s ({rate * TICKS_TO_MM:.1f} mm/s)")

    if not args.apply:
        print("\npreview only — nothing moved. Re-run with --apply.")
        mc.disconnect()
        return 0

    print("\nPaying cable OUT with no tension is what unspools the rollers.")
    print("Keep light tension on the wire by hand, and stop with Ctrl+C if it snags.")
    if input("Type 'go' to move: ").strip().lower() != "go":
        print("aborted — nothing moved.")
        mc.disconnect()
        return 1

    period = 1.0 / args.hz
    try:
        while True:
            done = True
            for m in targets:
                d = args.target - current[m]
                if abs(d) > args.step:
                    current[m] += args.step if d > 0 else -args.step
                    done = False
                else:
                    current[m] = args.target
                mc.set_position(m, current[m])
            print(f"  {' '.join(f'{m}:{current[m]}' for m in targets)}   ",
                  end="\r", flush=True)
            if done:
                break
            time.sleep(period)
    except KeyboardInterrupt:
        print("\ninterrupted — motors left where they are")
    finally:
        time.sleep(0.3)
        print(f"\n\nfinal: {mc.get_positions()}")
        mc.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
