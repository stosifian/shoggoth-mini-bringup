"""TEST A — read-only: what does Present_Position actually report?

ZERO RISK. Torque is disabled and no position is ever commanded. You turn the spool
by hand; this only samples registers.

WHY THIS IS FIRST. Every guard built this week assumed Present_Position is a plain
integer in a predictable range. Observed instead: 6101, 7340, and -1548. Vendor
documentation says 0-4095 (community register reference), while the product page
advertises multi-turn +/-7 turns — and the code runs the servos with both angle
limits set to 0, which that reference calls MOTOR MODE, where Goal_Position is
documented as speed/direction rather than position. Nothing about the register
semantics in this configuration is established. This measures it.

WHAT IT ANSWERS
  * does the reading stay within 0..4095, or exceed it?
  * does it ever go negative?
  * where exactly does it wrap, and by how much?
  * is a full mechanical revolution 4096 ticks of reading?

HOW TO RUN IT
  1. Detach the tendon, or accept that you are turning a loaded spool by hand.
  2. Start the tool. Torque goes off; the spool should turn freely.
  3. Turn the spool SLOWLY by hand: about two full turns one way, then two back.
     Slowly matters — the wrap detector assumes sampling outpaces motion.
  4. Ctrl+C (or wait for --duration) to stop. CSV and plot are written.

  python tools/char_a_readonly.py --motor 2
  python tools/char_a_readonly.py --motor 2 --duration 60 --hz 50
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.configs import get_hardware_config  # noqa: E402
from shoggoth_mini.hardware.motors import MotorController  # noqa: E402
from shoggoth_mini.common.constants import MOTOR_NAMES  # noqa: E402
from char_common import Recorder, render_position_trace, summarise_position, col, load  # noqa: E402

FIELDS = ["t", "motor", "present", "goal", "speed", "load", "torque",
          "min_limit", "max_limit", "mode"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="shoggoth_mini/configs/default_hardware.yaml")
    ap.add_argument("--motor", default="2", choices=MOTOR_NAMES)
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--csv", default="diagnostics/char_a.csv")
    ap.add_argument("--plot", default="diagnostics/char_a.png")
    ap.add_argument("--keep-torque", action="store_true",
                    help="do NOT disable torque (default is to disable it so the "
                         "spool turns freely)")
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

    # Static configuration, recorded once so the trace is interpretable later.
    static = {k: rd(k) for k in
              ("Min_Angle_Limit", "Max_Angle_Limit", "Mode", "Torque_Enable")}
    print(f"\nmotor {m} configuration as found:")
    for k, v in static.items():
        print(f"    {k:<18} {v}")
    if static["Min_Angle_Limit"] == 0 and static["Max_Angle_Limit"] == 0:
        print("    -> both angle limits 0: the community register reference calls")
        print("       this MOTOR MODE, in which Goal_Position is documented as")
        print("       speed/direction, not position. Noted, not assumed.")

    if not args.keep_torque:
        try:
            bus.write("Torque_Enable", 0, m)
            print(f"\ntorque DISABLED on motor {m} — spool should turn freely")
        except Exception as e:
            print(f"could not disable torque: {e}")

    print(f"\nsampling at {args.hz:.0f} Hz for up to {args.duration:.0f}s.")
    print("Turn the spool SLOWLY by hand: ~2 turns one way, then ~2 back.")
    print("Ctrl+C to finish early.\n")

    rec = Recorder(args.csv, FIELDS)
    period = 1.0 / args.hz
    last_print = 0.0
    try:
        t_end = time.time() + args.duration
        while time.time() < t_end:
            present = rd("Present_Position")
            rec.log(motor=m, present=present, goal=rd("Goal_Position"),
                    speed=rd("Present_Speed"), load=rd("Present_Load"),
                    torque=rd("Torque_Enable"),
                    min_limit=static["Min_Angle_Limit"],
                    max_limit=static["Max_Angle_Limit"], mode=static["Mode"])
            now = time.time()
            if now - last_print > 0.25:
                print(f"  present={present}   samples={rec.rows}    ",
                      end="\r", flush=True)
                last_print = now
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        rec.close()
        mc.disconnect()

    rows = load(args.csv)
    stats = summarise_position(col(rows, "present"))
    print(f"\n\n=== TEST A RESULT — motor {m}, {rec.rows} samples ===")
    for k, v in stats.items():
        print(f"    {k:<18} {v}")

    print("\ninterpretation:")
    if stats.get("samples", 0) < 10:
        print("    too few samples — nothing to conclude")
    else:
        if stats.get("negative_samples", 0):
            print("    * reading DOES go negative — so it is not an unsigned 0..4095 field")
        else:
            print("    * no negative readings in this run")
        if stats.get("above_period", 0):
            print("    * reading EXCEEDS 4095 — multi-turn accumulation, not a single turn")
        else:
            print("    * reading stayed below 4096")
        if stats.get("wraps", 0):
            print(f"    * {stats['wraps']} wrap(s) detected — it folds rather than accumulating")
        else:
            print("    * no wraps detected")

    render_position_trace(args.csv, args.plot,
                          title=f"Test A (read-only) — motor {m}, torque off",
                          cmd_key="goal")
    print(f"csv  -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
