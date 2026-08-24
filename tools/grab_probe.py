"""Find how far the grab pose can actually go, by stepping magnitude up slowly.

Upstream grab is |c| = 0.70 (2867 ticks, 77 mm of cable). On this build that is far
past the measured useful range and past max_travel_ticks, so it was being clamped —
and clamping DISTORTS the gesture, because the three motors clip by different
amounts and the bend direction shifts. This walks the magnitude up in small steps so
you can find where the tentacle stops gaining reach, or starts to bind.

THE OBJECTIVE SIGNAL: after each step it reports position error — commanded target
minus where the servo actually got to. A tendon that binds stalls the servo, so
error climbs. That is the limit, independent of judgement:

    error stays near zero  -> the mechanism is following, keep going
    error starts climbing  -> binding; that magnitude is past the limit

DELIBERATELY EXCEEDS max_travel_ticks. It raises the limit in-process only, prompts
before every step, returns to rest between steps, and never goes past the upstream
0.70. Nothing persists: the config file is untouched.

  python tools/grab_probe.py
  python tools/grab_probe.py --start 0.25 --stop 0.55 --step 0.05
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.configs import get_hardware_config  # noqa: E402
from shoggoth_mini.hardware.motors import MotorController  # noqa: E402
from shoggoth_mini.control.geometry import cursor_to_motor_positions  # noqa: E402
from shoggoth_mini.common.constants import (  # noqa: E402
    MOTOR_NAMES, MOTOR_NORMALIZED_POSITIONS,
)

TICKS_TO_MM = 0.11 / 4096 * 1000
UPSTREAM_MAX = 0.70
# A real stall is SUSTAINED; a single large sample is a transient (scheduling
# hiccup, direction reversal). Measured 2026-08-13: normal running lag is 10-16
# ticks, so a 30-tick single-sample threshold false-triggered on a 32-tick blip.
STALL_TICKS = 60          # lag that counts as "not following"
STALL_SAMPLES = 5         # consecutive samples over the threshold before aborting


def grab_cursor(magnitude: float) -> np.ndarray:
    """The grab direction (motor 2's axis), scaled. Direction is preserved."""
    return MOTOR_NORMALIZED_POSITIONS["2"] * magnitude


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="shoggoth_mini/configs/default_hardware.yaml")
    ap.add_argument("--start", type=float, default=0.25)
    ap.add_argument("--stop", type=float, default=UPSTREAM_MAX)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--ramp-ticks", type=int, default=6,
                    help="ticks per ramp increment (small = slow)")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--log", default="diagnostics/grab_probe.csv",
                    help="CSV of every commanded step and the position reached. "
                         "Written continuously, so a run that ends in a bang still "
                         "leaves a record of what was commanded when.")
    args = ap.parse_args()

    if args.stop > UPSTREAM_MAX:
        raise SystemExit(f"--stop capped at the upstream value {UPSTREAM_MAX}")

    cfg = get_hardware_config(args.config)
    original_limit = int(getattr(cfg, 'max_travel_ticks', 0) or 0)  # guard reverted; may be absent
    mc = MotorController(cfg)
    mc.connect()
    calib = mc.get_calibration_data()
    if not calib:
        raise SystemExit("no calibration loaded")

    # Raise the limit IN THIS PROCESS ONLY so the probe can exceed it deliberately.
    peak_needed = max(
        abs(int(cursor_to_motor_positions(
            cursor_pos=grab_cursor(args.stop), calibrated_ticks_map=calib)[0][m]) - calib[m])
        for m in MOTOR_NAMES
    )
    if original_limit:
        cfg.max_travel_ticks = peak_needed + 50

    print(f"\ncalibration: {calib}")
    if original_limit:
        print(f"travel limit raised {original_limit} -> {cfg.max_travel_ticks} "
              f"FOR THIS PROCESS ONLY (config file untouched)")
    else:
        print("NOTE: no travel guard is configured (reverted to upstream), so this "
              "tool has nothing to override — motion is unbounded. The modulo-4096 "
              "reversal is NOT guarded; keep every commanded step under 2048 ticks.")
    print(f"stepping |c| {args.start} -> {args.stop} in {args.step} increments\n")
    print("Watch the tentacle. Stop when it stops gaining reach, starts to bow, or")
    print("the position error column climbs — that is the tendon binding.\n")

    logf = open(args.log, "w", newline="")
    logw = csv.writer(logf)
    logw.writerow(["t", "phase", "magnitude",
                   "cmd_1", "cmd_2", "cmd_3",
                   "act_1", "act_2", "act_3",
                   "err_1", "err_2", "err_3"])
    t0 = time.time()

    def log_row(phase, mag, cmd):
        try:
            act = {m: mc.get_position(m) for m in MOTOR_NAMES}
        except Exception:
            act = {m: "" for m in MOTOR_NAMES}
        row = [f"{time.time()-t0:.3f}", phase, f"{mag:.3f}"]
        row += [cmd.get(m, "") for m in MOTOR_NAMES]
        row += [act.get(m, "") for m in MOTOR_NAMES]
        row += [("" if act.get(m) == "" else act[m] - cmd.get(m, 0))
                for m in MOTOR_NAMES]
        logw.writerow(row)
        logf.flush()          # flush every row: the interesting run is the one that
                              # ends unexpectedly

    def ramp_to(cursor, mag=float("nan"), phase="ramp"):
        """Ramp to a cursor pose, ABORTING if the servos stop keeping up.

        Why the abort matters: a naive ramp advances its own commanded counter
        regardless of where the motor actually is. If a tendon binds, the servo
        stalls while the command keeps marching ahead — and when the obstruction
        releases, the servo is suddenly free with a target hundreds of ticks away
        and slams to it at full speed. That is the "sudden wind" heard at |c|=0.55
        on 2026-08-13: stored command error discharging, not the ramp itself.

        So each step re-reads the actual position and stops if the gap exceeds
        STALL_TICKS. The command can never run away from the mechanism.
        """
        target, _ = cursor_to_motor_positions(
            cursor_pos=np.asarray(cursor, dtype=float), calibrated_ticks_map=calib
        )
        target = {m: int(target[m]) for m in MOTOR_NAMES}
        current = {m: mc.get_position(m) for m in MOTOR_NAMES}
        over = 0
        while True:
            deltas = {m: target[m] - current[m] for m in MOTOR_NAMES}
            if all(abs(d) <= args.ramp_ticks for d in deltas.values()):
                mc.set_positions(target)
                break
            for m in MOTOR_NAMES:
                current[m] += max(-args.ramp_ticks, min(args.ramp_ticks, deltas[m]))
            mc.set_positions(current)
            log_row(phase, mag, current)

            # Do not let the command outrun the mechanism.
            actual = {m: mc.get_position(m) for m in MOTOR_NAMES}
            lag = {m: current[m] - actual[m] for m in MOTOR_NAMES}
            worst_m = max(lag, key=lambda k: abs(lag[k]))
            over = over + 1 if abs(lag[worst_m]) > STALL_TICKS else 0
            if over >= STALL_SAMPLES:
                print(f"\n    ABORTING RAMP: motor {worst_m} has been >{STALL_TICKS} "
                      f"ticks behind for {over} consecutive steps "
                      f"(now {abs(lag[worst_m])}, "
                      f"{abs(lag[worst_m])*TICKS_TO_MM:.1f} mm) — sustained stall.")
                print("    Backing the command off to where the motor actually is,")
                print("    so nothing is stored up to discharge when the load frees.")
                mc.set_positions(actual)
                log_row(phase + "_aborted", mag, actual)
                time.sleep(0.3)
                return actual

            time.sleep(1.0 / args.hz)
        time.sleep(0.5)
        log_row(phase + "_settled", mag, target)
        return target

    mags = []
    m = args.start
    while m <= args.stop + 1e-9:
        mags.append(round(m, 3))
        m += args.step

    try:
        for mag in mags:
            cur = grab_cursor(mag)
            preview, _ = cursor_to_motor_positions(
                cursor_pos=cur, calibrated_ticks_map=calib
            )
            peak = max(abs(int(preview[m]) - calib[m]) for m in MOTOR_NAMES)
            print(f"--- |c| = {mag:.2f}   peak {peak} ticks ({peak*TICKS_TO_MM:.1f} mm)"
                  f"{'   [past the normal limit]' if peak > original_limit else ''}")
            if input("    Enter to go, 'q' to stop: ").strip().lower() == "q":
                break

            target = ramp_to(cur, mag, 'out')
            actual = mc.get_positions()
            err = {m: actual[m] - target[m] for m in MOTOR_NAMES}
            worst = max(abs(v) for v in err.values())
            verdict = ("FOLLOWING" if worst < STALL_TICKS
                       else "NOT FOLLOWING — binding/stalling")
            print(f"    position error {err}  worst {worst} ticks "
                  f"({worst*TICKS_TO_MM:.1f} mm)  -> {verdict}")
            input("    Enter to return to rest: ")
            ramp_to(np.array([0.0, 0.0]), mag, 'back')
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        print("\nreturning to rest...")
        try:
            ramp_to(np.array([0.0, 0.0]), float('nan'), 'final')
            print("final:", {m: mc.get_position(m) - calib[m] for m in MOTOR_NAMES},
                  "ticks from zero")
        except Exception as e:
            print("could not return to rest:", e)
        if original_limit:
            cfg.max_travel_ticks = original_limit
        mc.disconnect()
        logf.close()
        print(f"travel limit restored to {original_limit}")
        print(f"log written to {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
