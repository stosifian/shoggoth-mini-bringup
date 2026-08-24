"""TEST C — position-mode step response: what does the motor do with Goal_Position?

WHY. Every primitive, sweep, the idle loop and the closed loop command POSITION
(Mode 0, Goal_Position); only `calibrate` uses wheel mode. So the position command
space is the one that actually matters, and it is the one that is unmeasured.

Two specific things need numbers rather than inference:

  1. THE FOLD THRESHOLD. Tests this week inferred that these servos take the
     modulo-4096 SHORTEST PATH, so a command asking for more than ~2048 ticks
     travels the other way by (4096 - delta). That was inferred from three data
     points (+2100 -> -1995, +2662 -> -1434, +4096 -> 0). The guardband is being
     built on it, so it deserves a proper sweep with the threshold bracketed.

  2. THE RESPONSE PROFILE. Nothing knows how fast a position step actually moves,
     whether it overshoots, or how long it takes to settle. `closed_loop.py`
     clamps targets to zero+[-4000,+3000] but places NO limit on the step from the
     current position, so one frame can legitimately ask for ~4000 ticks. Whether
     that is survivable depends on numbers this test produces.

WHAT IT ANSWERS
  * commanded delta vs ACHIEVED signed travel, across the fold
  * exactly where the direction reverses (is it 2048, or 2047, or elsewhere?)
  * peak velocity, overshoot and settling time as a function of step size
  * whether the servo's own motion profile is trapezoidal or bang-bang

HOW IT IS SAFE
  * TENDON MUST BE DETACHED. Large steps are the point of this test.
  * the ladder runs SMALLEST FIRST, so the first surprise is the mildest.
  * every step is followed by a return to home, and the return delta is always
    the negative of what was ACHIEVED — which, if shortest-path holds, is never
    more than 2048. The return therefore cannot itself fold.
  * --dry-run prints the whole ladder with predicted outcomes and moves nothing.
  * torque is left OFF at the end, and no position is pinned on exit: reading a
    position and writing it back as a target is what caused an abrupt move on
    2026-08-17.

  python tools/char_c_step.py --motor 2 --dry-run
  python tools/char_c_step.py --motor 2
  python tools/char_c_step.py --motor 2 --deltas 50,200,500 --settle 1.0
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
from char_common import Recorder, load, col, unwrap_ticks  # noqa: E402

FIELDS = ["t", "motor", "phase", "step_index", "commanded_delta", "target",
          "home", "present", "present_speed", "load"]
SERVO_MODE = 0
PERIOD = 4096
TICKS_TO_MM = 0.11 / PERIOD * 1000

DEFAULT_LADDER = "50,200,500,1000,1500,1800,2000,2048,2100,2500,3000,4000"


def shortest_path(delta: int, period: int = PERIOD) -> int:
    """Signed travel predicted if the servo takes the modulo-period short arc."""
    return ((delta + period // 2) % period) - period // 2


def analyse(ts: np.ndarray, unw: np.ndarray, tol: float) -> dict:
    """Step-response metrics from one settle window, relative to its first sample."""
    ok = ~np.isnan(unw)
    if ok.sum() < 5:
        return {}
    t, x = ts[ok], unw[ok] - unw[ok][0]

    tail = max(1, len(x) // 5)
    final = float(np.mean(x[-tail:]))

    # settling time: last moment the trace is outside the tolerance band
    outside = np.abs(x - final) > tol
    settle = float(t[np.max(np.nonzero(outside))] ) if outside.any() else 0.0

    # overshoot measured in the direction of travel, so sign does not confuse it
    direction = np.sign(final) if final != 0 else 1.0
    peak = float(np.max(x * direction))
    overshoot = max(0.0, peak - abs(final))

    dt = np.diff(t)
    dt[dt <= 0] = np.nan
    vel = np.abs(np.diff(x) / dt)
    peak_vel = float(np.nanmax(vel)) if len(vel) else 0.0

    return dict(achieved=final, settle_s=settle, overshoot=overshoot,
                peak_vel=peak_vel, samples=int(ok.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="shoggoth_mini/configs/default_hardware.yaml")
    ap.add_argument("--motor", default="2", choices=MOTOR_NAMES)
    ap.add_argument("--deltas", default=DEFAULT_LADDER,
                    help="commanded step magnitudes in ticks, smallest first")
    ap.add_argument("--max-delta", type=int, default=4096,
                    help="refuse any commanded magnitude above this")
    ap.add_argument("--settle", type=float, default=1.5,
                    help="seconds to observe after each command")
    ap.add_argument("--hz", type=float, default=100.0)
    ap.add_argument("--tol", type=float, default=20.0,
                    help="settling band in ticks")
    ap.add_argument("--home", type=int, default=None,
                    help="move here first (in 500-tick increments) and use it as "
                         "home; pick mid-range so the ladder stays inside 0..4095")
    ap.add_argument("--allow-out-of-range", action="store_true",
                    help="run even if the ladder would command a target outside "
                         "0..4095 — see the note in the source before using this")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the ladder and predictions; move nothing")
    ap.add_argument("--csv", default="diagnostics/char_c.csv")
    ap.add_argument("--plot", default="diagnostics/char_c.png")
    args = ap.parse_args()

    mags = [int(x) for x in args.deltas.split(",") if x.strip()]
    too_big = [m for m in mags if abs(m) > args.max_delta]
    if too_big:
        print(f"refusing magnitudes above --max-delta {args.max_delta}: {too_big}")
        return 1
    mags = sorted(set(abs(m) for m in mags))

    # + then - at each magnitude, ascending: the first surprise is the mildest.
    ladder = [d * s for d in mags for s in (+1, -1)]

    print(f"\nladder for motor {args.motor} — {len(ladder)} steps, "
          f"each followed by a return to home")
    print(f"{'#':>3}{'commanded':>11}{'predicted travel':>19}{'':>4}")
    print("-" * 40)
    for i, d in enumerate(ladder):
        p = shortest_path(d)
        flag = "  <-- REVERSES" if np.sign(p) != np.sign(d) else ""
        print(f"{i:>3}{d:>11}{p:>19}{flag}")
    print(f"\nestimated motion time ~{len(ladder) * args.settle * 2:.0f}s")

    if args.dry_run:
        print("\ndry run — nothing moved.")
        return 0

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

    home = rd("Present_Position")
    print(f"\nmotor {m} present position = {home}")
    print(f"mode as found = {rd('Mode')}, torque = {rd('Torque_Enable')}")
    if home is None:
        print("could not read position — aborting")
        mc.disconnect()
        return 1

    # Test C (2026-08-17) established that step size is not the hazard: targets
    # inside 0..4095 track exactly at any step size, while a target ABOVE 4095
    # stops the motor at the ceiling and one BELOW 0 can trigger a multi-turn
    # runaway. So the ladder must be centred where it cannot leave the range.
    reach = max(mags)
    lo, hi = home - reach, home + reach
    if not (0 <= lo and hi <= PERIOD - 1):
        want = PERIOD // 2
        print(f"\n!! home {home} +/- {reach} spans [{lo}, {hi}], which leaves "
              f"0..{PERIOD - 1}.")
        print(f"   Targets outside that range do NOT track — this is what "
              f"corrupted the negative half of the first run.")
        if args.home is None and not args.allow_out_of_range:
            print(f"   Re-run with --home {want} (or a smaller --deltas ladder), "
                  f"or pass --allow-out-of-range to proceed anyway.")
            mc.disconnect()
            return 1

    if args.home is not None and args.home != home:
        if not (0 <= args.home <= PERIOD - 1):
            print(f"--home must be within 0..{PERIOD - 1}")
            mc.disconnect()
            return 1
        print(f"\nmoving to home {args.home} in 500-tick increments "
              f"(both endpoints in range, so every increment is in range)")
        bus.write("Mode", SERVO_MODE, m)
        bus.write("Torque_Enable", 1, m)
        time.sleep(0.2)
        while abs(args.home - home) > 2:
            home += int(np.clip(args.home - home, -500, 500))
            with mc._bus_lock:
                bus.write("Goal_Position", home, m)
            time.sleep(0.4)
            home = rd("Present_Position")
            print(f"  at {home}   ", end="\r", flush=True)
        print(f"\nhome = {home}")

    print("\n*** THE TENDON ON THIS MOTOR MUST BE DETACHED ***")
    print(f"Steps up to {max(mags)} ticks ({max(mags) * TICKS_TO_MM:.0f} mm of cable) "
          f"will be commanded.")
    if input("Type 'free' to continue: ").strip().lower() != "free":
        print("aborted — nothing moved.")
        mc.disconnect()
        return 1

    rec = Recorder(args.csv, FIELDS)
    period = 1.0 / args.hz
    results = []

    def observe(target, commanded_delta, phase, index):
        """Command one target, sample the whole settle window, return metrics."""
        samples = []
        t0 = time.time()
        with mc._bus_lock:
            bus.write("Goal_Position", int(target), m)
        while time.time() - t0 < args.settle:
            p = rd("Present_Position")
            samples.append((time.time() - t0, p))
            rec.log(motor=m, phase=phase, step_index=index,
                    commanded_delta=commanded_delta, target=int(target), home=home,
                    present=p, present_speed=rd("Present_Speed"),
                    load=rd("Present_Load"))
            time.sleep(period)

        ts = np.array([s[0] for s in samples], dtype=float)
        ps = np.array([np.nan if s[1] is None else s[1] for s in samples],
                      dtype=float)
        return analyse(ts, unwrap_ticks(ps, PERIOD), args.tol), ps

    try:
        # Mode first, torque after: the Mode write clears Torque_Enable.
        bus.write("Mode", SERVO_MODE, m)
        bus.write("Torque_Enable", 1, m)
        time.sleep(0.2)

        for i, d in enumerate(ladder):
            start = rd("Present_Position")
            if start is None:
                print(f"\nstep {i}: lost the position read — stopping")
                break

            met, _ = observe(start + d, d, f"step/{d:+d}", i)
            if not met:
                print(f"\nstep {i}: too few samples — stopping")
                break

            pred = shortest_path(d)
            met.update(commanded=d, predicted=pred, start=start,
                       error_vs_pred=met["achieved"] - pred)
            results.append(met)
            print(f"  step {i:>2}: cmd {d:>+6}  ->  achieved {met['achieved']:>+8.0f}"
                  f"  (predicted {pred:>+6})   peak {met['peak_vel']:>6.0f} ticks/s"
                  f"   settle {met['settle_s']:.2f}s"
                  f"   overshoot {met['overshoot']:>4.0f}")

            # Return home. The delta is the negative of what was ACHIEVED, so if
            # shortest-path holds this is never more than 2048 and cannot fold.
            observe(home, 0, f"return/{d:+d}", i)

    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        try:
            bus.write("Torque_Enable", 0, m)
            print("\ntorque OFF, no position pinned. Mode left at servo.")
        except Exception as e:
            print(f"cleanup issue: {e}")
        rec.close()
        mc.disconnect()

    if not results:
        print("no usable steps")
        return 1

    print(f"\n=== TEST C RESULT — motor {m} ===")
    print(f"{'cmd':>8}{'achieved':>10}{'predicted':>11}{'err':>8}"
          f"{'peak t/s':>10}{'mm/s':>8}{'settle':>9}{'over':>7}")
    print("-" * 71)
    for r in results:
        print(f"{r['commanded']:>8}{r['achieved']:>10.0f}{r['predicted']:>11}"
              f"{r['error_vs_pred']:>8.0f}{r['peak_vel']:>10.0f}"
              f"{r['peak_vel'] * TICKS_TO_MM:>8.1f}{r['settle_s']:>9.2f}"
              f"{r['overshoot']:>7.0f}")

    print("\ninterpretation:")

    reversed_steps = [r for r in results
                      if np.sign(r["achieved"]) != np.sign(r["commanded"])
                      and abs(r["achieved"]) > args.tol]
    if reversed_steps:
        smallest = min(abs(r["commanded"]) for r in reversed_steps)
        largest_ok = max([abs(r["commanded"]) for r in results
                          if r not in reversed_steps] or [0])
        print(f"    * DIRECTION REVERSES. Smallest commanded magnitude that went "
              f"backwards: {smallest}")
        print(f"      Largest that tracked correctly: {largest_ok}")
        print(f"      -> the fold threshold is bracketed in ({largest_ok}, {smallest}]")
    else:
        print("    * no step reversed direction in this ladder — either the ladder "
              "stops short of the threshold, or shortest-path does not hold")

    errs = np.array([abs(r["error_vs_pred"]) for r in results])
    print(f"    * error vs the modulo-{PERIOD} shortest-path prediction: "
          f"median {np.median(errs):.0f}, worst {errs.max():.0f} ticks")
    if errs.max() <= max(3 * args.tol, 60):
        print("      -> shortest-path predicts every step. The model is confirmed.")
    else:
        print("      -> some steps deviate from the prediction; inspect the plot.")

    vmax = max(r["peak_vel"] for r in results)
    print(f"    * peak velocity observed {vmax:.0f} ticks/s "
          f"({vmax * TICKS_TO_MM:.0f} mm/s of cable)")
    print(f"    * slowest settle {max(r['settle_s'] for r in results):.2f}s, "
          f"worst overshoot {max(r['overshoot'] for r in results):.0f} ticks")

    _plot(args.csv, args.plot, results, m, args)
    print(f"csv  -> {args.csv}")
    return 0


def _plot(csv_path, out_path, results, motor, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load(csv_path)
    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1, 1, 1.2])
    fig.suptitle(f"Test C — position-mode step response, motor {motor}", fontsize=13)

    cmd = np.array([r["commanded"] for r in results], dtype=float)
    ach = np.array([r["achieved"] for r in results], dtype=float)

    # panel 1: the headline — commanded vs achieved, against both models
    ax = fig.add_subplot(gs[0, :])
    span = np.linspace(cmd.min() * 1.05 - 50, cmd.max() * 1.05 + 50, 2000)
    ax.plot(span, span, ls=":", lw=1.2, color="gray", label="ideal (achieved = commanded)")
    ax.plot(span, [shortest_path(int(round(s))) for s in span], lw=1.2,
            color="tab:orange", label=f"modulo-{PERIOD} shortest path")
    ax.plot(cmd, ach, "o", ms=7, color="tab:blue", label="measured")
    ax.axvline(PERIOD // 2, color="tab:red", lw=1, ls="--", alpha=.6)
    ax.axvline(-PERIOD // 2, color="tab:red", lw=1, ls="--", alpha=.6,
               label=f"±{PERIOD // 2} (predicted threshold)")
    ax.axhline(0, color="gray", lw=0.8, alpha=.5)
    ax.set_xlabel("commanded delta (ticks)")
    ax.set_ylabel("achieved signed travel (ticks)")
    ax.set_title("points on the dotted line track the command; points on the orange "
                 "sawtooth took the short arc the other way round", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    # panel 2: overlaid step traces, so the motion profile itself is visible
    ax = fig.add_subplot(gs[1, :])
    norm = plt.Normalize(0, max(abs(cmd)) or 1)
    cmap = plt.get_cmap("viridis")
    for r in results:
        d = r["commanded"]
        sel = [row for row in rows if row.get("phase") == f"step/{d:+d}"]
        if len(sel) < 3:
            continue
        # Recorder's `t` is time since the RUN started, not since this command;
        # re-zero per phase or every trace lands at a different x offset.
        t = col(sel, "t")
        x = unwrap_ticks(col(sel, "present"), PERIOD)
        finite = ~np.isnan(x)
        if finite.sum() < 3:
            continue
        ax.plot(t[finite] - t[finite][0], x[finite] - x[finite][0], lw=1.2,
                color=cmap(norm(abs(d))))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label="|commanded delta| (ticks)", pad=0.01)
    ax.axhline(0, color="gray", lw=0.8, alpha=.5)
    ax.set_xlabel("time since command (s)")
    ax.set_ylabel("travel from step start (ticks)")
    ax.set_title("step responses overlaid — slope is velocity, flat tail is settled; "
                 "traces that dive negative folded", fontsize=10)
    ax.grid(alpha=.3)

    # panel 3: velocity vs step size — does the servo profile, or just go flat out?
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(np.abs(cmd), [r["peak_vel"] for r in results], "o", color="tab:purple")
    ax.set_xlabel("|commanded delta| (ticks)")
    ax.set_ylabel("peak velocity (ticks/s)")
    ax.set_title("peak velocity vs step size", fontsize=10)
    ax.grid(alpha=.3)
    sec = ax.secondary_yaxis("right", functions=(lambda t: t * TICKS_TO_MM,
                                                 lambda s: s / TICKS_TO_MM))
    sec.set_ylabel("mm/s of cable")

    # panel 4: settling time and overshoot — the numbers a rate limit needs
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(np.abs(ach), [r["settle_s"] for r in results], "o",
            color="tab:green", label="settling time")
    ax.set_xlabel("|achieved travel| (ticks)")
    ax.set_ylabel("settling time (s)", color="tab:green")
    ax.grid(alpha=.3)
    ax2 = ax.twinx()
    ax2.plot(np.abs(ach), [r["overshoot"] for r in results], "s", ms=5,
             color="tab:red", label="overshoot")
    ax2.set_ylabel("overshoot (ticks)", color="tab:red")
    ax.set_title(f"settling (±{args.tol:.0f} ticks) and overshoot", fontsize=10)

    ax = fig.add_subplot(gs[3, :])
    ax.plot(cmd, ach-cmd, color="tab:green")
    ax.set_xlabel("commanded delta (ticks)")
    ax.set_ylabel("error: achieved - cmd (ticks)")
    ax.set_title("Error vs Target", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    print(f"plot -> {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
