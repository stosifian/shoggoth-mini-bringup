"""Adopt the motors' CURRENT positions as the calibrated zero.

Use after hand-tensioning: get the tentacle standing how you want it, then run this
to declare that pose the rest position. Equivalent to what `calibrate` saves at the
moment you press Enter, without going through the wheel-mode keyboard tool.

Nothing moves. It only rewrites the calibration file, and backs up the previous one
first (--restore puts it back).

Everything downstream is relative to these numbers — the cursor->tendon geometry,
the +/-max_travel_ticks limit, the RL policy's actuator lengths — so this is the
reference the whole robot hangs off.

  python tools/set_zero.py            # preview
  python tools/set_zero.py --apply
  python tools/set_zero.py --restore
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.configs import get_hardware_config  # noqa: E402
from shoggoth_mini.hardware.motors import MotorController  # noqa: E402
from shoggoth_mini.hardware.calibration import save_calibration  # noqa: E402
from shoggoth_mini.common.constants import MOTOR_NAMES  # noqa: E402

TICKS_TO_MM = 0.11 / 4096 * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="shoggoth_mini/configs/default_hardware.yaml")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--restore", action="store_true",
                    help="restore the backup taken by the last --apply")
    args = ap.parse_args()

    cfg = get_hardware_config(args.config)
    calib_path = Path(cfg.calibration_file)
    backup_path = calib_path.with_suffix(".pre-setzero.json")

    if args.restore:
        if not backup_path.exists():
            print(f"no backup at {backup_path}")
            return 1
        shutil.copy(backup_path, calib_path)
        print(f"restored {calib_path} from {backup_path}")
        return 0

    old = {}
    try:
        raw = json.load(open(calib_path))
        old = {m: v["ticks"] if isinstance(v, dict) else int(v) for m, v in raw.items()}
    except Exception:
        print("(no existing calibration to compare against)")

    mc = MotorController(cfg)
    mc.connect()
    now = mc.get_positions()
    mc.disconnect()

    print(f"\n{'motor':>6}{'old zero':>10}{'new zero':>10}{'change':>9}{'cable':>11}")
    print("-" * 47)
    for m in MOTOR_NAMES:
        o = old.get(m)
        d = "" if o is None else f"{now[m] - o:+d}"
        mm = "" if o is None else f"{(now[m] - o) * TICKS_TO_MM:+.1f}mm"
        print(f"{m:>6}{('—' if o is None else o):>10}{now[m]:>10}{d:>9}{mm:>11}")

    if old:
        spread = max(now[m] - old[m] for m in MOTOR_NAMES) - \
                 min(now[m] - old[m] for m in MOTOR_NAMES)
        print(f"\nspread of the change across tendons: {spread} ticks "
              f"({spread * TICKS_TO_MM:.1f} mm)")

    if not args.apply:
        print("\npreview only — calibration unchanged. Re-run with --apply.")
        return 0

    print("\nThis makes the CURRENT pose the rest position. It should be the pose you")
    print("want the robot to return to: upright, all three tendons taut.")
    if input("Type 'go' to write: ").strip().lower() != "go":
        print("aborted — calibration unchanged.")
        return 1

    if calib_path.exists():
        shutil.copy(calib_path, backup_path)
        print(f"backed up previous calibration -> {backup_path}")
    save_calibration(now, calib_path)
    print(f"wrote {calib_path}: {now}")
    print("\nNext: tools/tendon_sweep.py --magnitude 0.18 --spokes 6 --cycles 2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
