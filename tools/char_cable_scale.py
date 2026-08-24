"""CABLE SCALE — how many mm of cable is one tick, actually?

WHY. Every safety number in this project is quoted in mm of cable, and every one
of them is derived from a single assumed constant: 0.11 m per spool rotation,
i.e. 0.02686 mm/tick. That constant has never been measured on this build. If it
is off by 20%, so is every limit computed from it, and so is the interpretation of
every speed in the characterisation series.

It can only be measured cleanly while the tendon is THREADED BUT UNTIED — the wire
must be on the spool to see it move, but must not be anchored, or the tentacle
deflects and the wire path length changes as you measure. That is the state the
robot is in now, and it will not come back once the knots are tied.

WHAT IT ANSWERS
  * mm of cable per tick, measured, with a residual against the assumption
  * whether the ratio is constant across the travel range or varies with how
    much wire is already wound on the spool (it should vary a little: the
    effective radius grows as the wire builds up)
  * whether winding in and paying out give the same scale (hysteresis)

HOW TO RUN IT
  1. Park the motor at mid-range: python tools/park_motors.py --motors N --apply
  2. Put a mark on the wire where it enters the tentacle guide, and set a ruler
     alongside it. A fine marker or a scrap of tape both work.
  3. Start this tool. For each step it moves a known number of ticks slowly,
     then asks you how far the mark travelled in mm.
  4. Enter the measurement. Blank or 's' skips a step.

  Sign convention: enter the DISTANCE, always positive. The tool knows which
  direction it commanded.

HOW IT IS SAFE
  * every target is clamped to 0..4095 and the run is symmetric about the park
    point, so no net cable is paid out
  * moves are ramped at the same slow rate park_motors uses (~4 mm/s)
  * Ctrl+C stops between steps

  python tools/char_cable_scale.py --motor 2
  python tools/char_cable_scale.py --motor 2 --deltas 500,1000,1500
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.configs import get_hardware_config  # noqa: E402
from shoggoth_mini.hardware.motors import MotorController  # noqa: E402
from shoggoth_mini.common.constants import MOTOR_NAMES  # noqa: E402
from char_common import Recorder  # noqa: E402

FIELDS = ["t", "motor", "step", "direction", "commanded_ticks", "actual_ticks",
          "measured_mm", "mm_per_tick"]
PERIOD = 4096
ASSUMED_M_PER_ROTATION = 0.11
ASSUMED_MM_PER_TICK = ASSUMED_M_PER_ROTATION / PERIOD * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="shoggoth_mini/configs/default_hardware.yaml")
    ap.add_argument("--motor", default="2", choices=MOTOR_NAMES)
    ap.add_argument("--deltas", default="500,1000,1500",
                    help="tick moves to measure, each done in both directions")
    ap.add_argument("--park", type=int, default=2048)
    ap.add_argument("--step", type=int, default=3, help="ticks per increment")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--csv", default="diagnostics/char_cable.csv")
    ap.add_argument("--plot", default="diagnostics/char_cable.png")
    args = ap.parse_args()

    deltas = sorted(int(d) for d in args.deltas.split(",") if d.strip())
    reach = max(deltas)
    if not (0 <= args.park - reach and args.park + reach <= PERIOD - 1):
        raise SystemExit(f"park {args.park} +/- {reach} leaves 0..{PERIOD - 1}; "
                         f"use a smaller --deltas or a different --park")

    rate = args.step * args.hz
    print(f"\ncable scale — motor {args.motor}")
    print(f"  assumption on file: {ASSUMED_M_PER_ROTATION} m/rotation "
          f"= {ASSUMED_MM_PER_TICK:.5f} mm/tick")
    print(f"  steps: {deltas} ticks, each wound IN and paid OUT")
    print(f"  predicted travel at the assumed scale: " +
          ", ".join(f"{d} ticks -> {d * ASSUMED_MM_PER_TICK:.1f} mm" for d in deltas))
    print(f"  ramp rate {rate:.0f} ticks/s = {rate * ASSUMED_MM_PER_TICK:.1f} mm/s")
    print("\n  THE TENDON MUST BE THREADED BUT UNTIED.")
    print("  Mark the wire where it enters the guide and set a ruler alongside.")
    if input("\nType 'go' to begin: ").strip().lower() != "go":
        print("aborted — nothing moved.")
        return 1

    mc = MotorController(get_hardware_config(args.config))
    mc.connect()
    m = args.motor

    def pos():
        return mc.get_position(m)

    def ramp_to(target):
        """Slow ramp, same rate park_motors uses. Target is clamped in range."""
        target = int(np.clip(target, 0, PERIOD - 1))
        cur = pos()
        while abs(target - cur) > args.step:
            cur += args.step if target > cur else -args.step
            mc.set_position(m, cur)
            time.sleep(1.0 / args.hz)
        mc.set_position(m, target)
        time.sleep(0.3)
        return pos()

    rec = Recorder(args.csv, FIELDS)
    samples = []
    step_no = 0

    try:
        start = pos()
        if abs(start - args.park) > 5:
            print(f"\nparking: {start} -> {args.park}")
            start = ramp_to(args.park)
        print(f"parked at {start}\n")

        for d in deltas:
            for direction, label in ((+1, "wind IN"), (-1, "pay OUT")):
                step_no += 1
                before = pos()
                print(f"step {step_no}: {label} {d} ticks "
                      f"(predicted {d * ASSUMED_MM_PER_TICK:.1f} mm) — moving...")
                after = ramp_to(before + direction * d)
                actual = after - before
                print(f"  moved {actual:+d} ticks by the encoder")

                raw = input("  measured travel of the mark, in mm "
                            "(blank to skip): ").strip()
                measured = None
                if raw and raw.lower() not in ("s", "skip"):
                    try:
                        measured = abs(float(raw))
                    except ValueError:
                        print("  not a number — skipping this step")

                if measured is not None and actual != 0:
                    mm_per_tick = measured / abs(actual)
                    samples.append(dict(step=step_no, direction=direction,
                                        commanded=d, actual=actual,
                                        measured=measured, mm_per_tick=mm_per_tick))
                    print(f"  -> {mm_per_tick:.5f} mm/tick "
                          f"({100 * (mm_per_tick / ASSUMED_MM_PER_TICK - 1):+.1f}% "
                          f"vs assumption)")
                    rec.log(motor=m, step=step_no, direction=direction,
                            commanded_ticks=d, actual_ticks=actual,
                            measured_mm=measured, mm_per_tick=f"{mm_per_tick:.6f}")

                # Return to park so the next step starts from the same wind state.
                ramp_to(args.park)

    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        try:
            ramp_to(args.park)
            print(f"\nreturned to park {pos()}")
        except Exception as e:
            print(f"could not return to park: {e}")
        rec.close()
        mc.disconnect()

    if not samples:
        print("\nno measurements entered — nothing to fit")
        return 1

    ticks = np.array([abs(s["actual"]) for s in samples], dtype=float)
    mm = np.array([s["measured"] for s in samples], dtype=float)
    # Fit through the origin: zero ticks must be zero travel.
    fit = float(np.sum(ticks * mm) / np.sum(ticks * ticks))
    resid = mm - fit * ticks

    print(f"\n=== CABLE SCALE — motor {m}, {len(samples)} measurements ===")
    print(f"{'step':>6}{'dir':>9}{'ticks':>8}{'mm':>8}{'mm/tick':>10}{'resid mm':>10}")
    print("-" * 51)
    for s, r in zip(samples, resid):
        print(f"{s['step']:>6}{'wind IN' if s['direction'] > 0 else 'pay OUT':>9}"
              f"{abs(s['actual']):>8}{s['measured']:>8.1f}"
              f"{s['mm_per_tick']:>10.5f}{r:>10.2f}")

    print(f"\n    fitted scale      {fit:.5f} mm/tick")
    print(f"    assumption        {ASSUMED_MM_PER_TICK:.5f} mm/tick")
    print(f"    difference        {100 * (fit / ASSUMED_MM_PER_TICK - 1):+.1f}%")
    print(f"    implied spool     {fit * PERIOD / 1000:.4f} m/rotation "
          f"(assumed {ASSUMED_M_PER_ROTATION})")
    print(f"    residual RMS      {np.sqrt(np.mean(resid ** 2)):.2f} mm")

    ins = [s["mm_per_tick"] for s in samples if s["direction"] > 0]
    outs = [s["mm_per_tick"] for s in samples if s["direction"] < 0]
    print("\ninterpretation:")
    if abs(fit / ASSUMED_MM_PER_TICK - 1) < 0.05:
        print("    * the assumed 0.11 m/rotation holds within 5% — every mm figure "
              "in the series stands")
    else:
        print(f"    * the assumption is off by "
              f"{100 * (fit / ASSUMED_MM_PER_TICK - 1):+.0f}%. Every limit and speed "
              f"quoted in mm needs rescaling by {fit / ASSUMED_MM_PER_TICK:.3f}.")
    if ins and outs:
        h = abs(np.mean(ins) - np.mean(outs)) / fit
        print(f"    * wind-in vs pay-out differ by {100 * h:.1f}%"
              + ("  -> hysteresis worth accounting for" if h > 0.05 else
                 "  -> no meaningful hysteresis"))
    if len(set(abs(s["commanded"]) for s in samples)) > 1:
        big = [s for s in samples if abs(s["commanded"]) == max(deltas)]
        small = [s for s in samples if abs(s["commanded"]) == min(deltas)]
        if big and small:
            sb = np.mean([s["mm_per_tick"] for s in big])
            ss = np.mean([s["mm_per_tick"] for s in small])
            print(f"    * scale at {max(deltas)} ticks vs {min(deltas)} ticks: "
                  f"{100 * (sb / ss - 1):+.1f}% — a growing effective spool radius "
                  f"would show up here")

    _plot(args.plot, samples, fit, m)
    print(f"csv  -> {args.csv}")
    return 0


def _plot(out_path, samples, fit, motor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Cable scale — motor {motor}", fontsize=13)

    ticks = np.array([abs(s["actual"]) for s in samples], dtype=float)
    mm = np.array([s["measured"] for s in samples], dtype=float)
    span = np.linspace(0, ticks.max() * 1.1, 100)

    for d, c, mk, lab in ((+1, "tab:blue", "o", "wind IN"),
                          (-1, "tab:orange", "s", "pay OUT")):
        sel = [s for s in samples if s["direction"] == d]
        if sel:
            ax[0].plot([abs(s["actual"]) for s in sel],
                       [s["measured"] for s in sel], mk, ms=8, color=c, label=lab)
    ax[0].plot(span, fit * span, "-", color="tab:green", lw=1.6,
               label=f"fit {fit:.5f} mm/tick")
    ax[0].plot(span, ASSUMED_MM_PER_TICK * span, "--", color="gray", lw=1.4,
               label=f"assumed {ASSUMED_MM_PER_TICK:.5f}")
    ax[0].set_xlabel("encoder travel (ticks)")
    ax[0].set_ylabel("measured cable travel (mm)")
    ax[0].set_title("measured travel against both scales", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    ax[1].axhline(0, color="gray", lw=1)
    for d, c, mk, lab in ((+1, "tab:blue", "o", "wind IN"),
                          (-1, "tab:orange", "s", "pay OUT")):
        sel = [s for s in samples if s["direction"] == d]
        if sel:
            ax[1].plot([abs(s["actual"]) for s in sel],
                       [s["measured"] - fit * abs(s["actual"]) for s in sel],
                       mk, ms=8, color=c, label=lab)
    ax[1].set_xlabel("encoder travel (ticks)")
    ax[1].set_ylabel("residual (mm)")
    ax[1].set_title("residual against the fit — a trend with travel means the "
                    "effective spool radius changes", fontsize=10)
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130)
    print(f"plot -> {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
