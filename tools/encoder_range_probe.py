"""Map what a servo actually DOES with a Goal_Position past 4095.

Two observations on this build contradict each other under any simple model:

    commanded 4200  (105 past 4095)   -> read 4201       honoured
    commanded 5910  (1815 past 4095)  -> read ~1805      = 5910 - 4096

If the goal wrapped at 4096, the first would have read 104. If Present_Position
reported mod 4096, the first would have read 105. Neither happened. So there is a
discontinuity somewhere between, and nobody has measured where.

This steps the goal upward past the boundary and records where the motor lands,
producing the actual transfer function instead of another plausible story.

REQUIREMENTS — read before running:
  * TENDON DETACHED on the motor under test. This deliberately commands large
    travel; if the wire is attached it may pay out or wind in ~110 mm.
  * Bypasses both the travel limit and the encoder clamp, because those are the
    very things under investigation. It is the only tool that does.

  python tools/encoder_range_probe.py --motor 2
  python tools/encoder_range_probe.py --motor 2 --start 4000 --stop 6200 --step 200
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.configs import get_hardware_config  # noqa: E402
from shoggoth_mini.hardware.motors import MotorController  # noqa: E402
from shoggoth_mini.common.constants import MOTOR_NAMES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="shoggoth_mini/configs/default_hardware.yaml")
    ap.add_argument("--motor", default="2", choices=MOTOR_NAMES)
    ap.add_argument("--start", type=int, default=3900)
    ap.add_argument("--stop", type=int, default=6200)
    ap.add_argument("--step", type=int, default=200)
    ap.add_argument("--settle", type=float, default=1.2)
    args = ap.parse_args()

    mc = MotorController(get_hardware_config(args.config))
    mc.connect()
    m = args.motor
    bus = mc._motor_bus
    start_pos = mc.get_position(m)

    print(f"\nmotor {m} currently at {start_pos}")
    print("\n*** THE TENDON ON THIS MOTOR MUST BE DETACHED ***")
    print("This commands large travel and bypasses both the travel limit and the")
    print("encoder clamp — those are what is being measured.\n")
    if input("Type 'detached' to continue: ").strip().lower() != "detached":
        print("aborted — nothing moved.")
        mc.disconnect()
        return 1

    def read(field):
        v = bus.read(field, m)
        return int(v.item() if hasattr(v, "item") else v[0])

    print(f"\n{'commanded':>10}{'goal readback':>15}{'present':>10}"
          f"{'present-cmd':>13}   interpretation")
    print("-" * 74)

    rows = []
    try:
        for cmd in range(args.start, args.stop + 1, args.step):
            # Straight to the bus: bypasses _limit_travel and the 0..4095 clamp.
            with mc._bus_lock:
                bus.write("Goal_Position", int(cmd), m)
            time.sleep(args.settle)
            goal_rb = read("Goal_Position")
            present = read("Present_Position")
            diff = present - cmd

            if abs(diff) <= 30:
                note = "honoured"
            elif abs(diff + 4096) <= 30:
                note = "WRAPPED (-4096)"
            elif abs(diff - 4096) <= 30:
                note = "WRAPPED (+4096)"
            else:
                note = f"other ({diff:+d})"
            print(f"{cmd:>10}{goal_rb:>15}{present:>10}{diff:>13}   {note}")
            rows.append((cmd, goal_rb, present, diff, note))
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        print(f"\nreturning to {start_pos} ...")
        with mc._bus_lock:
            bus.write("Goal_Position", int(start_pos), m)
        time.sleep(1.5)
        print(f"now at {mc.get_position(m)}")
        mc.disconnect()

    honoured = [r for r in rows if r[4] == "honoured"]
    wrapped = [r for r in rows if "WRAPPED" in r[4]]
    if honoured and wrapped:
        print(f"\nDISCONTINUITY: honoured up to {max(r[0] for r in honoured)}, "
              f"wrapped from {min(r[0] for r in wrapped)}")
    elif honoured and not wrapped:
        print(f"\nAll commands honoured up to {max(r[0] for r in honoured)} — "
              f"no wrap found in this range.")
    elif wrapped and not honoured:
        print("\nEverything wrapped — the boundary is below --start.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
