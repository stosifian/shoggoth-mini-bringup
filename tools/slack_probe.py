"""Measure per-tendon slack from motor load telemetry.

!!! DOES NOT WORK ON THIS HARDWARE — kept only to document the dead end. !!!

Tested 2026-08-10 on STS3215 (model 777):
  * Present_Load (60) is quantised to steps of 8 and never rises with tendon pull.
    Motor 1 read 0 at +818 ticks (22 mm) of pull, having read 8 at +134.
  * Present_Current (69) reads a flat 0 at every offset up to 736 ticks. Not
    populated by this firmware.
Both channels sit below the torque needed to bend a soft TPU tentacle, so any
"tension onset" this reports is the first random non-zero sample — noise, not a
measurement. It once claimed a 273-tick spread between tendons on pure noise.

Use the empirical loop instead: retension.py in increments, then tendon_sweep.py,
judging whether the 60deg and 120deg poses are visibly distinct.

Original description follows.
--------------------------------------------------------------------------------
Measure per-tendon slack without opening the dome, using motor load telemetry.

The problem this solves: once the dome is closed you cannot watch the marker lines,
and the encoder sits UPSTREAM of the horn, so it reports success whether the tendon
moved or not. But torque is observable. A tendon under tension makes its motor work;
a slack one does not.

Method: pull each tendon along its own axis in small increments, reading Present_Load
at each step. Load stays near zero while the motor is only reeling in slack, then
climbs once the tendon starts actually bending the tentacle. The breakpoint IS the
slack, in ticks.

Reading the result:
  * all three break at a similar offset      -> evenly tensioned, healthy
  * one breaks much later than the others    -> that tendon carries extra slack
                                                (retension it specifically)
  * load rises then COLLAPSES mid-ramp       -> something let go: horn slip on the
                                                spline, or a knot pulling through
  * load never rises at all                  -> that tendon is not transmitting

Tendon pull axes come from constants.py: motor 1 = [0,+1] (90deg),
motor 2 = [0.866,-0.5] (-30deg), motor 3 = [-0.866,-0.5] (210deg).

  python tools/slack_probe.py
  python tools/slack_probe.py --max-magnitude 0.20 --steps 24
  python tools/slack_probe.py --motors 3
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
PULL_ANGLE_DEG = {"1": 90.0, "2": -30.0, "3": 210.0}
MAX_STEP = 10
# Tension onset is detected RELATIVE to each motor's own unloaded baseline rather
# than against a fixed constant: Present_Load is PWM-derived and its noise floor
# differs per motor, so any absolute threshold is a guess. Onset = the first step
# where load exceeds baseline by this margin and stays up.
ONSET_MARGIN = 6


def decode_load(raw):
    """Feetech Present_Load is sign-magnitude: bit 10 is direction.

    The bus hands back a numpy array, not a scalar — coerce before masking.
    """
    if hasattr(raw, "item"):
        raw = raw.item() if raw.size == 1 else raw[0]
    return int(raw) & 0x3FF


def glide(mc, positions, step=MAX_STEP, hz=100.0):
    current = {m: mc.get_position(m) for m in MOTOR_NAMES}
    while True:
        deltas = {m: positions[m] - current[m] for m in MOTOR_NAMES}
        if all(abs(d) <= step for d in deltas.values()):
            mc.set_positions(positions)
            break
        for m in MOTOR_NAMES:
            current[m] += max(-step, min(step, deltas[m]))
        mc.set_positions(current)
        time.sleep(1.0 / hz)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="shoggoth_mini/configs/default_hardware.yaml")
    ap.add_argument("--max-magnitude", type=float, default=0.18)
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--motors", default="1,2,3")
    ap.add_argument("--margin", type=int, default=ONSET_MARGIN,
                    help="load rise above baseline counted as tension onset")
    args = ap.parse_args()

    if args.max_magnitude > 0.25:
        print("refusing: above the measured useful range (~0.25)")
        return 1

    targets = [m.strip() for m in args.motors.split(",") if m.strip()]
    cfg = get_hardware_config(args.config)
    mc = MotorController(cfg)
    mc.connect()
    calib = mc.get_calibration_data()
    if not calib or all(v == 0 for v in calib.values()):
        print("no calibration loaded — run `calibrate` first")
        return 1

    bus = mc._motor_bus

    def load_of(motor, samples=5):
        """Median of several reads — Present_Load is PWM-derived and noisy."""
        vals = []
        for _ in range(samples):
            vals.append(decode_load(bus.read("Present_Load", motor)))
            time.sleep(0.02)
        return sorted(vals)[len(vals) // 2]

    print(f"\ncalibration: {calib}")
    print(f"ramping to |c|={args.max_magnitude} in {args.steps} steps, "
          f"onset = baseline + {args.margin}\n")

    summary = {}
    try:
        for motor in targets:
            angle = np.radians(PULL_ANGLE_DEG[motor])
            glide(mc, {m: calib[m] for m in MOTOR_NAMES})
            time.sleep(0.3)
            baseline = load_of(motor)

            print(f"=== motor {motor}  (pull axis {PULL_ANGLE_DEG[motor]:+.0f}deg, "
                  f"unloaded baseline {baseline}) ===")
            print(f"{'|c|':>7} {'offset':>8} {'mm':>7} {'load':>7} {'rise':>6}")
            print("-" * 42)

            breakpoint_ticks = None
            collapsed = False
            peak_load = 0

            for i in range(1, args.steps + 1):
                mag = args.max_magnitude * i / args.steps
                cursor = np.array([mag * np.cos(angle), mag * np.sin(angle)])
                target, _ = cursor_to_motor_positions(
                    cursor_pos=cursor, calibrated_ticks_map=calib
                )
                target = {m: int(target[m]) for m in MOTOR_NAMES}
                glide(mc, target)
                time.sleep(0.15)

                offset = mc.get_position(motor) - calib[motor]
                load = load_of(motor)
                rise = load - baseline
                peak_load = max(peak_load, load)

                mark = ""
                if breakpoint_ticks is None and rise >= args.margin:
                    breakpoint_ticks = offset
                    mark = "  <-- tension onset"
                if breakpoint_ticks is not None and (peak_load - baseline) > args.margin * 3 \
                        and rise < args.margin // 2:
                    collapsed = True
                    mark = "  <-- LOAD COLLAPSE (slip?)"

                print(f"{mag:>7.3f} {offset:>+8} {offset * TICKS_TO_MM:>+6.2f} "
                      f"{load:>7} {rise:>+6}{mark}")

            summary[motor] = (breakpoint_ticks, peak_load, collapsed)
            glide(mc, calib)
            time.sleep(0.3)
            print()
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        glide(mc, calib)

        print("=" * 58)
        print(f"{'motor':>6} {'slack (ticks)':>15} {'mm':>8} {'peak load':>11}")
        print("-" * 58)
        for motor, (bp, peak, collapsed) in summary.items():
            if collapsed:
                note = "  LOAD COLLAPSED — suspect slip"
            elif bp is None:
                note = ("  no load rise — either slack beyond this range, or "
                        "below Present_Load's resolution")
            else:
                note = ""
            bp_s = "n/a" if bp is None else str(bp)
            mm_s = "n/a" if bp is None else f"{bp * TICKS_TO_MM:.2f}"
            print(f"{motor:>6} {bp_s:>15} {mm_s:>8} {peak:>11}{note}")

        vals = [bp for bp, _, _ in summary.values() if bp is not None]
        if len(vals) > 1:
            spread = max(vals) - min(vals)
            print(f"\nspread between tendons: {spread} ticks "
                  f"({spread * TICKS_TO_MM:.2f} mm)")
            print("Large spread -> retension the laggard with:")
            print("  tools/retension.py --ticks <spread> --motors <n> --apply")
        mc.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
