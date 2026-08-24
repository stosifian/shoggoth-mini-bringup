"""Bench check for the Feetech motor bus — confirms all IDs are on the chain.

Exercises the project's real MotorController path: loads the hardware config,
connects (which verifies every configured motor ID is present on the bus), and
reads back positions. Use it after ID assignment, after assembly, and on
hardware day to confirm the full daisy-chained bus before driving anything.

Prereq: motors daisy-chained to the driver board, 12 V power on, USB connected.
Run from the shoggoth-mini dir, venv active:

  python tools/motor_bus_check.py
  python tools/motor_bus_check.py --config shoggoth_mini/configs/default_hardware.yaml

Pass  -> "All motors found" + a positions dict for every ID.
Fail  -> empty/echoed port means the wrong config; a "Not all motors found" /
         serial error means a broken chain link, power, USB, or an unset ID.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.configs import get_hardware_config  # noqa: E402
from shoggoth_mini.hardware.motors import MotorController  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="shoggoth_mini/configs/default_hardware.yaml",
                    help="hardware config yaml (must set the real serial port)")
    args = ap.parse_args()

    cfg = get_hardware_config(args.config)
    print(f"using port: {cfg.port}")
    if not cfg.port:
        raise SystemExit("config has an empty port — check the yaml / --config path")

    mc = MotorController(cfg)
    try:
        mc.connect()  # raises if any configured motor ID is missing on the bus
        positions = mc.get_positions()
        print("positions:", {k: int(np_to_scalar(v)) for k, v in positions.items()})
        print("OK: full motor chain present and readable.")
    finally:
        try:
            mc.disconnect()
        except Exception:
            pass


def np_to_scalar(v):
    """get_positions returns numpy arrays; coerce to a plain int for printing."""
    try:
        return v.item() if hasattr(v, "item") else v[0]
    except Exception:
        return v


if __name__ == "__main__":
    main()
