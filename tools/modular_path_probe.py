"""Test whether the servo moves by MODULAR SHORTEST PATH, and find where it flips.

HYPOTHESIS (2026-08-14): the servo treats Goal_Position modulo 4096 and travels the
shorter of the two arcs. So the movement actually performed is

    signed_move = ((target - current + 2048) mod 4096) - 2048

which means ANY commanded move larger than 2048 ticks executes in the OPPOSITE
direction, by (4096 - |delta|) ticks.

If true it explains, without any other assumption:
  * grab commanding +2662 and the motor going -1434 (tearing the wire off)
  * a "return to 3245" from 6101 travelling +1239 to 7340
  * a 1.0-turn spin check (+4096) not moving at all and reporting -4096 error
  * a +204 command being honoured normally

PREDICTIONS THIS TEST CHECKS
    delta +1900  -> moves +1900   (under the half-turn boundary)
    delta +2100  -> moves -1996   (over it: 4096 - 2100)
    delta -2100  -> moves +1996
The sign flip should occur sharply at 2048.

REQUIREMENTS
  * TENDON DETACHED on the motor under test — this commands ~2000 tick moves.
  * Bypasses the travel limit and encoder clamp deliberately; they are downstream
    of the behaviour being characterised.

  python tools/modular_path_probe.py --motor 2
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.configs import get_hardware_config  # noqa: E402
from shoggoth_mini.hardware.motors import MotorController  # noqa: E402
from shoggoth_mini.common.constants import MOTOR_NAMES  # noqa: E402

DEFAULT_DELTAS = [300, 1000, 1800, 1950, 2000, 2100, 2300, 3000, -1800, -2100, -3000]


def predicted(delta: int) -> int:
    """Movement the modular-shortest-path hypothesis predicts."""
    return ((delta + 2048) % 4096) - 2048


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="shoggoth_mini/configs/default_hardware.yaml")
    ap.add_argument("--motor", default="2", choices=MOTOR_NAMES)
    ap.add_argument("--deltas", default=",".join(str(d) for d in DEFAULT_DELTAS))
    ap.add_argument("--settle", type=float, default=1.6)
    args = ap.parse_args()

    deltas = [int(x) for x in args.deltas.split(",") if x.strip()]

    mc = MotorController(get_hardware_config(args.config))
    mc.connect()
    m = args.motor
    bus = mc._motor_bus
    base = mc.get_position(m)

    print(f"\nmotor {m} base position {base}")
    print("\n*** TENDON ON THIS MOTOR MUST BE DETACHED — commands up to ~3000 ticks ***")
    if input("Type 'detached' to continue: ").strip().lower() != "detached":
        print("aborted — nothing moved.")
        mc.disconnect()
        return 1

    def goto(target):
        with mc._bus_lock:
            bus.write("Goal_Position", int(target), m)
        time.sleep(args.settle)
        return mc.get_position(m)

    def safe_return(target):
        """Walk back in sub-2048 hops, since a big jump would itself go modular."""
        for _ in range(6):
            cur = mc.get_position(m)
            gap = target - cur
            if abs(gap) <= 20:
                return cur
            hop = max(-1500, min(1500, gap))
            goto(cur + hop)
        return mc.get_position(m)

    print(f"\n{'delta':>7}{'predicted':>11}{'actual':>9}{'match':>8}   note")
    print("-" * 60)
    hits = misses = 0
    try:
        for d in deltas:
            cur = mc.get_position(m)
            actual_pos = goto(cur + d)
            moved = actual_pos - cur
            pred = predicted(d)
            ok = abs(moved - pred) <= 40
            hits, misses = (hits + 1, misses) if ok else (hits, misses + 1)
            note = ("as commanded" if pred == d
                    else f"REVERSED (commanded {d:+d})")
            print(f"{d:>7}{pred:>11}{moved:>9}{'  yes' if ok else '   NO':>8}   {note}")
            safe_return(base)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        print("\nreturning to base...")
        final = safe_return(base)
        print(f"at {final} (base {base})")
        mc.disconnect()

    print(f"\n{hits} match, {misses} mismatch")
    if misses == 0 and hits:
        print("HYPOTHESIS HOLDS: the servo moves by modular shortest path.")
        print("Practical rule: never command a move larger than ~2048 ticks;")
        print("beyond that it travels the other way.")
    elif hits == 0:
        print("HYPOTHESIS FAILS — movement does not follow modular shortest path.")
    else:
        print("MIXED — the model is incomplete; look at which deltas disagree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
