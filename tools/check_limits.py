"""Post-retension check: can anything the robot does now leave 0..4095?

Run this after EVERY retension. Retensioning moves the calibrated zeros, and
because every commanded position is an offset from a zero, moving them changes
how much headroom each motion has. The zeros only ever move UP (retension is
always wind-in to take up slack), so the margin that shrinks is the one toward
4095 — and `set_position` refuses anything past it, which stops a primitive
mid-motion or kills the idle thread.

Checks every path that can command a position:

  * all motion primitives, by running the real primitive code through a
    recording stub
  * idle, from the breathing pattern's magnitude bound
  * the closed-loop policy, at its configured magnitude cap
  * the closed loop's own per-motor safety clamp

Static and hardware-free: it reads the calibration file and computes. Nothing
moves, nothing connects, so it is safe to run at any time.

  python tools/check_limits.py
  python tools/check_limits.py --noise 0.010 --trials 500   # include noise
  python tools/check_limits.py --offset 200                 # what-if: zeros +200

Exit code is 0 if everything fits, 1 if anything would be refused.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import shoggoth_mini.control.primitives as prims  # noqa: E402
from shoggoth_mini.configs import get_control_config, get_hardware_config  # noqa: E402
from shoggoth_mini.control.geometry import cursor_to_motor_positions  # noqa: E402
from shoggoth_mini.common.constants import MOTOR_NAMES  # noqa: E402

PERIOD = 4096
TICKS_TO_MM = 0.11 / PERIOD * 1000
OVERSHOOT = 36          # measured in the step-response test; targets land this far past


def worst_target(magnitude, calib, samples=180):
    """Highest and lowest target over every direction at this cursor magnitude."""
    hi, lo = -10 ** 9, 10 ** 9
    for a in np.linspace(0, 2 * np.pi, samples, endpoint=False):
        t, _ = cursor_to_motor_positions(
            cursor_pos=np.array([magnitude * np.cos(a), magnitude * np.sin(a)]),
            calibrated_ticks_map=calib,
        )
        hi = max(hi, max(t.values()))
        lo = min(lo, min(t.values()))
    return lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hardware-config",
                    default="shoggoth_mini/configs/default_hardware.yaml")
    ap.add_argument("--control-config",
                    default="shoggoth_mini/configs/default_control.yaml")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="noise_scale to apply to primitives (orchestrator uses "
                         "motion_noise_scale, the CLI defaults to 0.010)")
    ap.add_argument("--trials", type=int, default=1,
                    help="repeat the primitive pass N times; only useful with --noise")
    ap.add_argument("--offset", type=int, default=0, metavar="TICKS",
                    help="what-if: add this to every zero without changing the file")
    args = ap.parse_args()

    hw = get_hardware_config(args.hardware_config)
    cc = get_control_config(args.control_config)
    raw = json.load(open(hw.calibration_file))
    calib = {k: int(v["ticks"] if isinstance(v, dict) else v)
             for k, v in raw.items() if k in MOTOR_NAMES}
    if args.offset:
        calib = {m: v + args.offset for m, v in calib.items()}

    print(f"\ncalibration: {calib}" + (f"   (+{args.offset} what-if)" if args.offset else ""))
    print(f"headroom up: " + ", ".join(
        f"m{m} {PERIOD - 1 - v}" for m, v in sorted(calib.items())))
    print(f"headroom dn: " + ", ".join(f"m{m} {v}" for m, v in sorted(calib.items())))

    failures = []
    print(f"\n{'what':>26}{'lowest':>9}{'highest':>9}{'margin':>9}{'mm':>7}   verdict")
    print("-" * 76)

    def report(label, lo, hi, note=""):
        margin = (PERIOD - 1) - hi
        bad = hi > PERIOD - 1 or lo < 0
        if bad:
            failures.append(label)
        print(f"{label:>26}{lo:>9.0f}{hi:>9.0f}{margin:>9.0f}"
              f"{margin * TICKS_TO_MM:>7.1f}   "
              f"{'*** WOULD BE REFUSED ***' if bad else ('ok' + (' ' + note if note else ''))}")

    # --- primitives, through the real primitive code -----------------------
    import time as _t
    real_sleep = _t.sleep
    _t.sleep = lambda *a, **k: None
    prims.time.sleep = lambda *a, **k: None
    try:
        from char_primitive_sweep import RecordingController, run_one, BEHAVIOUR
        for key in [k for k in BEHAVIOUR if k != "sweep"]:
            lo, hi = 10 ** 9, -10 ** 9
            for _ in range(max(1, args.trials)):
                rec = RecordingController(calib)
                run_one(rec, key, calib, args.noise)
                for _t_, cmd in rec.commands:
                    for v in cmd.values():
                        hi = max(hi, v)
                        lo = min(lo, v)
            if hi > -10 ** 8:
                report(key, lo, hi)
    finally:
        _t.sleep = real_sleep
        prims.time.sleep = real_sleep

    # --- the tendon sweep, at its practical maximum ------------------------
    lo, hi = worst_target(0.25, calib)
    report("tendon_sweep |c|=0.25", lo, hi)

    # --- idle --------------------------------------------------------------
    pc = cc.idle_motion_pattern_config
    amp_max = float(pc["amplitude_range"][1])
    oav = float(pc.get("origin_avoidance_radius", 0.0))
    # The origin-avoidance offset is PERPENDICULAR to the oscillation, so the
    # magnitude bound is the Pythagorean sum, not the linear one.
    idle_mag = float(np.hypot(amp_max, oav))
    lo, hi = worst_target(idle_mag, calib)
    report(f"idle (|c|={idle_mag:.3f})", lo, hi)

    # --- closed loop --------------------------------------------------------
    cap = float(cc.max_2d_action_magnitude)
    lo, hi = worst_target(cap, calib)
    report(f"closed loop (cap {cap})", lo, hi)

    # --- the closed loop's own per-motor clamp ------------------------------
    clamp_lo = min(calib[m] + cc.safety_offset_min for m in calib)
    clamp_hi = max(calib[m] + cc.safety_offset_max for m in calib)
    report("safety_offset clamp band", clamp_lo, clamp_hi,
           note="(a backstop; should be inside the range)")

    # --- what grab could be -------------------------------------------------
    print()
    zero2 = calib["2"]
    for reserve in (0, 200):
        mx = (PERIOD - 1 - OVERSHOOT - reserve - zero2) / PERIOD
        tag = "arithmetic max" if reserve == 0 else "with 200 ticks spare"
        print(f"  grab magnitude {tag:>22}: {max(mx, 0):.3f}")
    cur = float(np.linalg.norm(np.asarray(prims.GRAB_CONFIG.grab_cursor_pos)))
    print(f"  {'currently set to':>39}: {cur:.3f}"
          f"{'   <-- TOO LARGE' if cur > (PERIOD - 1 - OVERSHOOT - zero2) / PERIOD else ''}")

    print()
    if failures:
        print(f"FAIL — {len(failures)} would be refused: {', '.join(failures)}")
        print("  Lower the offending magnitude, or bring the calibrated zeros down.")
        return 1
    print(f"PASS — every commanded position stays inside 0..{PERIOD - 1}.")
    print(f"  Tightest margin above is what fails first on the next retension.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
