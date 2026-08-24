"""Controlled spin test for individual motors — verifies the horn/roller drivetrain.

Commands a precise, ramped rotation on a chosen SUBSET of motors and reports the
encoder error. Use it after assembling each motor (horn + roller + tendon) and
before seating it in the base, so a bad joint is caught while it's still easy to
reach.

IMPORTANT — what this can and cannot tell you:

  * The magnetic encoder is on the motor's OUTPUT SHAFT, upstream of the horn.
    If the horn slips on the spline, the shaft still reaches its target and this
    script reports success. Software CANNOT detect horn slip.
  * So this tool's job is to deliver an exact, repeatable rotation. YOU verify
    mechanically: mark the horn and the roller with a marker line, run one full
    turn, and confirm the marks are still aligned and the roller turned exactly
    as far as commanded.
  * Slip only appears under torque. Apply light hand tension to the tendon while
    it winds — no tension, no slip, no information. Keep fingers clear of the
    roller/cover pinch point; tension the wire well away from the spool.

All configured motors must be present on the bus (connect() verifies the whole
chain), but only the ones named with --motors are commanded. The others are left
untouched, so a partially assembled build is fine.

Nothing moves until you confirm at the prompt. Positions are commanded RELATIVE
to wherever each motor currently sits, and the motor is returned to its start
position afterwards, so this does not disturb calibration.

  python tools/motor_spin_check.py --motors 1,2
  python tools/motor_spin_check.py --motors 1 --turns 0.5

NOTE: --turns must be < 1.0. Present_Position wraps at 4096, so a full turn reads
as no movement at all, and the return leg silently does nothing.
  python tools/motor_spin_check.py --motors 1,2 --cycles 3
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.common.constants import MOTOR_ONE_FULL_TURN_TICKS  # noqa: E402
from shoggoth_mini.configs import get_hardware_config  # noqa: E402
from shoggoth_mini.hardware.motors import MotorController  # noqa: E402

STEP_TICKS = 64  # ramp granularity — keeps the roller at a watchable speed
STEP_DELAY = 0.02  # seconds between steps (~0.8 rotations/sec)


def scalar(v):
    """get_position may return a numpy array; coerce to a plain int."""
    try:
        return int(v.item() if hasattr(v, "item") else v[0])
    except (AttributeError, IndexError, TypeError):
        return int(v)


def ramp_to(mc, motor, start, target):
    """Walk a motor from start to target in small steps so it moves visibly."""
    delta = target - start
    if delta == 0:
        return
    steps = max(1, abs(delta) // STEP_TICKS)
    for i in range(1, steps + 1):
        mc.set_position(motor, start + round(delta * i / steps))
        time.sleep(STEP_DELAY)
    mc.set_position(motor, target)
    time.sleep(0.15)  # let it settle before reading back


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config",
        default="shoggoth_mini/configs/default_hardware.yaml",
        help="hardware config yaml (must set the real serial port)",
    )
    ap.add_argument(
        "--motors",
        default="1",
        help="comma-separated motor names to spin, e.g. '1,2' (default: 1)",
    )
    ap.add_argument(
        "--turns", type=float, default=0.5,
        help="rotation per sweep (default: 0.5). MUST stay below 1.0 — see below."
    )
    ap.add_argument(
        "--cycles", type=int, default=1, help="out-and-back repetitions (default: 1)"
    )
    args = ap.parse_args()

    cfg = get_hardware_config(args.config)
    if not cfg.port:
        raise SystemExit("config has an empty port — check the yaml / --config path")

    motors = [m.strip() for m in args.motors.split(",") if m.strip()]
    unknown = [m for m in motors if m not in cfg.motor_config]
    if unknown:
        raise SystemExit(f"unknown motor(s) {unknown}; config has {list(cfg.motor_config)}")

    ticks = int(args.turns * MOTOR_ONE_FULL_TURN_TICKS)

    # A full turn winds ~110 mm of cable in, far beyond the tendon's travel, so the
    # motor stalls against the mechanism and never reaches the target. Observed
    # 2026-08-13 with the old default of 1.0: error came back as exactly -4096 and
    # the motor never unwound.
    #
    # (That exact -4096 originally looked like single-turn encoder wrap. Later
    # testing DISPROVED wrapping — Goal_Position is multi-turn and a command past
    # 4095 is honoured — so the likeliest reading is a mechanical stall, not an
    # encoder artefact. The guard stands either way: a full turn of wind-in is not
    # a safe thing to command on a threaded tendon.)
    if abs(ticks) >= MOTOR_ONE_FULL_TURN_TICKS:
        raise SystemExit(
            f"refusing --turns {args.turns}: {ticks} ticks is a full revolution or "
            f"more, which a single-turn encoder cannot represent. The motor would "
            f"wind in and never unwind. Use --turns 0.75 or less."
        )
    print(f"port:      {cfg.port}")
    print(f"spinning:  {motors}   (untouched: "
          f"{[m for m in cfg.motor_config if m not in motors]})")
    print(f"sweep:     {args.turns} turn = {ticks} ticks, {args.cycles} cycle(s)\n")
    print("Mark the horn AND the roller with a marker line before running.")
    print("Apply light tension to the tendon by hand — slip only shows under load.")
    print("Watch: does the roller rotate exactly as far as the shaft? Marks still aligned?\n")

    if input("Motors will MOVE. Type 'go' to continue: ").strip().lower() != "go":
        raise SystemExit("aborted — nothing moved.")

    mc = MotorController(cfg)
    try:
        mc.connect()  # verifies the whole chain; does NOT move anything
        for motor in motors:
            start = scalar(mc.get_position(motor))
            print(f"\nmotor {motor}: start={start}")
            worst = 0
            for c in range(1, args.cycles + 1):
                for label, target in (("out", start + ticks), ("back", start)):
                    ramp_to(mc, motor, scalar(mc.get_position(motor)), target)
                    actual = scalar(mc.get_position(motor))
                    err = actual - target
                    worst = max(worst, abs(err))
                    print(f"  cycle {c} {label:4}: target={target:7d} "
                          f"actual={actual:7d} err={err:+5d} ticks")
            tol = cfg.position_tolerance
            verdict = "OK" if worst <= tol else "OUT OF TOLERANCE"
            print(f"  worst |err| = {worst} ticks (tolerance {tol}) -> {verdict}")
            if worst > tol:
                print("  -> shaft did not track: check power, load, or a stalled motor.")
                print("     (This does NOT indicate horn slip — see the note above.)")
    except KeyboardInterrupt:
        print("\ninterrupted — motors left where they are.")
    finally:
        try:
            mc.disconnect()
        except Exception:
            pass

    print("\nEncoder tracking is only half the test. Now confirm mechanically:")
    print("  * horn/roller marker lines still aligned  -> no slip on the spline")
    print("  * roller rotated the full commanded amount -> horn is driving the spool")
    print("  * no wobble or rubbing through the rotation")


if __name__ == "__main__":
    main()
