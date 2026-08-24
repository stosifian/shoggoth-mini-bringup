"""Adjust tendon pretension by an exact tick count, without re-winding by hand.

`calibrate` is a hold-the-arrow-key wheel-mode tool — fine for taking up 80 cm of
slack, useless for a precise 150-tick nudge. But the calibration file is only three
numbers, so pretension can be changed exactly: command each motor to
`calibrated + ticks` in position mode, let the tentacle settle there, and save that
as the new rest pose.

Sign convention: `tick_sign: -1` in the hardware config means INCREASING ticks
SHORTEN the cable, so positive --ticks TIGHTENS. Applying the same delta to all three
raises pretension without biasing the rest pose in any direction.

Dry run by default — nothing moves and nothing is saved until you pass --apply.
The previous calibration is backed up next to the file before every write.

  python tools/retension.py --ticks 150              # preview
  python tools/retension.py --ticks 150 --apply      # tighten all three by 150
  python tools/retension.py --ticks -150 --apply     # loosen (undo)
  python tools/retension.py --ticks 100 --motors 3 --apply   # one tendon only
  python tools/retension.py --restore                # revert to the backup
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.configs import get_hardware_config  # noqa: E402
from shoggoth_mini.hardware.motors import MotorController  # noqa: E402
from shoggoth_mini.hardware.calibration import save_calibration  # noqa: E402
from shoggoth_mini.common.constants import MOTOR_NAMES  # noqa: E402

TICKS_TO_MM = 0.11 / 4096 * 1000
MAX_STEP = 10  # ticks per loop while gliding


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="shoggoth_mini/configs/default_hardware.yaml")
    ap.add_argument("--ticks", type=int, default=150,
                    help="positive tightens, negative loosens")
    ap.add_argument("--motors", default="1,2,3",
                    help="comma-separated motors to adjust (default all)")
    ap.add_argument("--apply", action="store_true", help="actually move and save")
    ap.add_argument("--restore", action="store_true",
                    help="restore the previous calibration backup and exit")
    args = ap.parse_args()

    cfg = get_hardware_config(args.config)
    calib_path = Path(cfg.calibration_file)
    backup_path = calib_path.with_suffix(".pre-retension.json")

    if args.restore:
        if not backup_path.exists():
            print(f"no backup at {backup_path}")
            return 1
        shutil.copy(backup_path, calib_path)
        print(f"restored {calib_path} from {backup_path}")
        print("power-cycle or re-run your control tool to load it")
        return 0

    targets = [m.strip() for m in args.motors.split(",") if m.strip()]
    if any(m not in MOTOR_NAMES for m in targets):
        print(f"unknown motor in {targets}; valid: {MOTOR_NAMES}")
        return 1

    if abs(args.ticks) > 600:
        print(f"refusing {args.ticks} ticks ({abs(args.ticks) * TICKS_TO_MM:.1f} mm) "
              f"in one step — retension incrementally and re-test between steps")
        return 1

    mc = MotorController(cfg)
    mc.connect()
    calib = mc.get_calibration_data()
    if not calib or all(v == 0 for v in calib.values()):
        print("no calibration loaded — run `calibrate` first")
        return 1

    new = {m: calib[m] + (args.ticks if m in targets else 0) for m in MOTOR_NAMES}

    print(f"\n{'motor':>6} {'current':>9} {'new':>9} {'delta':>8} {'cable':>10}")
    print("-" * 48)
    for m in MOTOR_NAMES:
        d = new[m] - calib[m]
        direction = "tighten" if d > 0 else ("loosen" if d < 0 else "")
        print(f"{m:>6} {calib[m]:>9} {new[m]:>9} {d:>+8} "
              f"{d * TICKS_TO_MM:>+7.2f} mm {direction}")

    if not args.apply:
        print("\ndry run — nothing moved, nothing saved. Re-run with --apply.")
        mc.disconnect()
        return 0

    # Glide there in rate-limited steps rather than jumping.
    print("\nmoving...")
    current = {m: mc.get_position(m) for m in MOTOR_NAMES}
    while True:
        deltas = {m: new[m] - current[m] for m in MOTOR_NAMES}
        if all(abs(d) <= MAX_STEP for d in deltas.values()):
            mc.set_positions(new)
            break
        for m in MOTOR_NAMES:
            current[m] += max(-MAX_STEP, min(MAX_STEP, deltas[m]))
        mc.set_positions(current)
        time.sleep(0.01)
    time.sleep(0.5)

    reached = {m: mc.get_position(m) for m in MOTOR_NAMES}
    err = {m: reached[m] - new[m] for m in MOTOR_NAMES}
    print(f"reached: {reached}   error: {err}")
    if max(abs(v) for v in err.values()) > 30:
        print("WARNING: motors did not reach target — not saving. Check for binding.")
        mc.disconnect()
        return 1

    shutil.copy(calib_path, backup_path)
    save_calibration(reached, calib_path)
    print(f"\nbacked up previous calibration -> {backup_path}")
    print(f"saved new calibration -> {calib_path}")
    print("\nNow re-run: tools/tendon_sweep.py --magnitude 0.18 --spokes 6 --cycles 2")
    print("Looking for: 60deg and 120deg poses visibly DIFFERENT from each other.")
    mc.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
