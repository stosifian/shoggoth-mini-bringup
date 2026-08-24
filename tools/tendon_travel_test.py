"""Exercise ONE motor across its full allowed travel, dome off, and watch the spool.

Built to verify a tendon after a wire-off-roller repair, before closing up. It drives
the two regimes that have actually caused failures on this build:

  WIND IN  (toward +max_travel_ticks) — tension rises. If the tentacle binds before
           the limit, tension spikes and the wire can be dragged over the flange into
           the horn gap. That is how motor 2's wire jammed on 2026-08-13.
  PAY OUT  (toward -max_travel_ticks) — tension falls. Slack wire has nothing holding
           it in the groove and can lift off. That is how the rollers were stripped
           during dome-clearance pay-out.

Both are invisible to the encoder: it sits upstream of the horn and the spool, so a
wire leaving the groove reads as a perfectly tracked command. YOUR EYES are the
instrument here; this tool just provides slow, bounded, interruptible motion and
catches the one thing it can — the servo failing to keep up.

SAFETY
  * ramps slowly, and ABORTS if the servo falls behind (a stalled motor with a
    command running ahead of it is what discharges violently when the load frees)
  * never exceeds max_travel_ticks; no in-process override, unlike grab_probe —
    that override is what broke the robot
  * pauses at every extreme so you can inspect before it moves on
  * returns to the calibrated zero on exit or Ctrl+C

WHAT TO WATCH
  * wire stays seated in the groove through the whole sweep, both directions
  * wire does not climb toward the flange as it winds in
  * marker line across shaft -> horn -> roller stays aligned (apply light hand
    tension to the tendon; slip only shows under load)
  * no clicking

  python tools/tendon_travel_test.py --motor 2
  python tools/tendon_travel_test.py --motor 2 --fraction 0.5 --cycles 2
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.configs import get_hardware_config  # noqa: E402
from shoggoth_mini.hardware.motors import MotorController  # noqa: E402
from shoggoth_mini.common.constants import MOTOR_NAMES  # noqa: E402

TICKS_TO_MM = 0.11 / 4096 * 1000
STALL_TICKS = 60
STALL_SAMPLES = 5


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="shoggoth_mini/configs/default_hardware.yaml")
    ap.add_argument("--motor", default="2", choices=MOTOR_NAMES)
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="fraction of max_travel_ticks to reach (default 1.0 = full)")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--step", type=int, default=4, help="ticks per increment")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--limit", type=int, default=1100,
                    help="travel in ticks either side of the zero (must be < 2048)")
    ap.add_argument("--log", default="diagnostics/tendon_travel.csv")
    args = ap.parse_args()

    cfg = get_hardware_config(args.config)
    # max_travel_ticks was reverted out of the config, so fall back to an explicit
    # default well under the 2048 modulo-4096 reversal boundary.
    base = int(getattr(cfg, "max_travel_ticks", 0) or args.limit)
    limit = int(base * max(0.0, min(1.0, args.fraction)))
    if limit <= 0:
        raise SystemExit("no travel limit available — pass --limit")
    if limit >= 2048:
        raise SystemExit(
            f"--limit {limit} is at or past the 2048 modulo-4096 reversal boundary; "
            f"a single command that large travels BACKWARDS. Use a smaller value."
        )

    mc = MotorController(cfg)
    mc.connect()
    calib = mc.get_calibration_data()
    m = args.motor
    zero = calib.get(m)
    if zero is None:
        raise SystemExit(f"no calibration for motor {m}")

    rate = args.step * args.hz
    print(f"\nmotor {m}: calibrated zero {zero}")
    print(f"travel  +/-{limit} ticks  (+/-{limit * TICKS_TO_MM:.1f} mm of cable)")
    print(f"speed   {rate:.0f} ticks/s ({rate * TICKS_TO_MM:.1f} mm/s), "
          f"stall abort at {STALL_TICKS} ticks")
    print("\nApply light tension to the tendon by hand. Watch the groove.\n")

    logf = open(args.log, "w", newline="")
    logw = csv.writer(logf)
    logw.writerow(["t", "phase", "cmd", "actual", "err"])
    t0 = time.time()

    def ramp(target, phase):
        cur = mc.get_position(m)
        over = 0
        while abs(target - cur) > args.step:
            cur += args.step if target > cur else -args.step
            mc.set_position(m, cur)
            act = mc.get_position(m)
            logw.writerow([f"{time.time()-t0:.3f}", phase, cur, act, act - cur])
            logf.flush()
            over = over + 1 if abs(act - cur) > STALL_TICKS else 0
            if over >= STALL_SAMPLES:
                print(f"\n  ABORT: motor is {abs(act-cur)} ticks "
                      f"({abs(act-cur)*TICKS_TO_MM:.1f} mm) behind the command — "
                      f"stalled. Backing off so nothing discharges.")
                mc.set_position(m, act)
                return False
            time.sleep(1.0 / args.hz)
        mc.set_position(m, target)
        time.sleep(0.4)
        act = mc.get_position(m)
        logw.writerow([f"{time.time()-t0:.3f}", phase + "_settled", target, act,
                       act - target])
        logf.flush()
        print(f"  reached {act} (target {target}, error {act-target:+d})")
        return True

    ok = True
    try:
        for c in range(args.cycles):
            print(f"=== cycle {c+1}/{args.cycles} ===")
            for phase, target in (("wind_in", zero + limit),
                                  ("to_zero", zero),
                                  ("pay_out", zero - limit),
                                  ("back_to_zero", zero)):
                label = {"wind_in": "WIND IN (tension rises — watch for the wire "
                                     "climbing the flange)",
                         "pay_out": "PAY OUT (goes slack — watch for the wire "
                                    "lifting out of the groove)",
                         "to_zero": "back to zero",
                         "back_to_zero": "back to zero"}[phase]
                print(f"\n  {label}")
                input("  Enter to move: ")
                if not ramp(target, phase):
                    ok = False
                    break
                if phase in ("wind_in", "pay_out"):
                    input("  Inspect the spool, then Enter to continue: ")
            if not ok:
                break
    except KeyboardInterrupt:
        print("\ninterrupted")
        ok = False
    finally:
        print("\nreturning to zero...")
        try:
            ramp(zero, "final")
        except Exception as e:
            print("could not return:", e)
        logf.close()
        mc.disconnect()
        print(f"\nlog: {args.log}")
        print("VERDICT:", "no stall detected — motor tracked the whole range"
              if ok else "stalled or interrupted — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
