"""One-shot readout of where the motors actually are. Read-only apart from torque.

Prints, per motor: present position, the calibrated zero, the offset between them in
ticks and mm of cable, and the ENCODER HEADROOM either side.

Headroom is the column that bites. Present_Position is a single-turn 0..4095
register, so a commanded target outside that range wraps and the servo takes the
long way round. On 2026-08-13 a zero at 3862 (233 ticks of room) met a routine +638
sweep, wrapped, and yanked 93 mm of cable in, seizing the tentacle. A full-range
cursor needs roughly 900 ticks of room on BOTH sides.

Also worth knowing: an offset near zero does NOT prove the tendons are right. The
encoder is upstream of the horn and the spool, so wire slipping on the roller or a
horn slipping on the spline both read as "at the calibrated position" while the
cable length has changed. Ticks are not cable length.

WARNING: connecting re-enables torque (MotorController.connect asserts it), so run
this BEFORE manual work on the mechanism, never after.

  python tools/motor_status.py
  python tools/motor_status.py --watch          # refresh until Ctrl+C
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.configs import get_hardware_config  # noqa: E402
from shoggoth_mini.hardware.motors import MotorController  # noqa: E402
from shoggoth_mini.common.constants import MOTOR_NAMES  # noqa: E402

TICKS_TO_MM = 0.11 / 4096 * 1000
HEADROOM_WARN = 900
ENCODER_MAX = 4095


def read_int(bus, field, motor):
    try:
        v = bus.read(field, motor)
        return int(v.item() if hasattr(v, "item") else v[0])
    except Exception:
        return None


def report(mc, calib):
    bus = mc._motor_bus
    now = mc.get_positions()
    # NOTE: the old "room up / room dn" columns (distance to 0 and 4095) were
    # removed. They rested on the belief that a target past 4095 wraps — DISPROVEN
    # 2026-08-13, these servos are multi-turn and honoured 4200. The column that
    # matters is travel used against max_travel_ticks, which is a real limit.
    limit = int(getattr(mc.config, "max_travel_ticks", 0) or 0)
    print(f"\n{'motor':>6}{'now':>7}{'calib':>7}{'offset':>8}{'cable':>11}"
          f"{'travel used':>13}{'torque':>8}{'mode':>6}")
    print("-" * 70)
    for m in MOTOR_NAMES:
        pos = now[m]
        zero = calib.get(m)
        off = "" if zero is None else f"{pos - zero:+d}"
        mm = "" if zero is None else f"{(pos - zero) * TICKS_TO_MM:+.1f}mm"
        if zero is None or limit <= 0:
            used = ""
        else:
            used = f"{100 * abs(pos - zero) / limit:.0f}% of ±{limit}"
        print(f"{m:>6}{pos:>7}{('—' if zero is None else zero):>7}{off:>8}{mm:>11}"
              f"{used:>13}"
              f"{str(read_int(bus, 'Torque_Enable', m)):>8}"
              f"{str(read_int(bus, 'Mode', m)):>6}")
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="shoggoth_mini/configs/default_hardware.yaml")
    ap.add_argument("--watch", action="store_true", help="refresh until Ctrl+C")
    ap.add_argument("--interval", type=float, default=0.5)
    args = ap.parse_args()

    cfg = get_hardware_config(args.config)
    calib = {}
    try:
        raw = json.load(open(cfg.calibration_file))
        calib = {m: v["ticks"] if isinstance(v, dict) else int(v)
                 for m, v in raw.items()}
    except Exception as e:
        print(f"(no usable calibration file: {e})")

    mc = MotorController(cfg)
    mc.connect()
    try:
        if not args.watch:
            report(mc, calib)
        else:
            print("watching — Ctrl+C to stop")
            while True:
                report(mc, calib)
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        mc.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
