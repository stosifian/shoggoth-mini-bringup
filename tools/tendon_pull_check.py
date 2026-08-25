"""Which tendons are actually transmitting force?

Winds each motor in ON ITS OWN by a small amount and watches Present_Load. A
tendon under tension resists, so load climbs as the motor pulls. Three failure
modes are distinguishable from that curve alone:

  load climbs, position tracks     -> healthy: the tendon is taut and pulling
  load stays flat, position tracks -> the motor turns but moves nothing. Either
                                      the tendon is slack, or the horn is
                                      slipping on the spline (the encoder sits
                                      UPSTREAM of the horn, so a slipping horn
                                      reports a perfect move while the spool
                                      never turns)
  load climbs immediately, no motion -> already hard against tension, or jammed

WHY ONE MOTOR AT A TIME. During a sweep all three move together, so a dead
tendon is masked: the other two still bend the tentacle, just not where you
asked. Isolating each one removes that confound.

WHY THIS IS SAFE WITH TENDONS ATTACHED
  * winds IN only (takes up cable, never pays out — paying out slack is what
    strips rollers)
  * small by default: 150 ticks, about 4 mm of cable
  * ramped at ~15 ticks per 20 ms, a tenth of the servo's ceiling
  * returns to the starting position after each motor
  * aborts a motor immediately if load exceeds --max-load
  * every target is inside 0..4095, so set_position's guard cannot trip

  python tools/tendon_pull_check.py                  # all three
  python tools/tendon_pull_check.py --motors 2       # just the suspect
  python tools/tendon_pull_check.py --ticks 250      # pull harder
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
from char_common import Recorder, load, col  # noqa: E402

FIELDS = ["t", "motor", "phase", "commanded", "present", "error", "load", "load_raw"]
PERIOD = 4096
TICKS_TO_MM = 0.11 / PERIOD * 1000
# Must exceed the servo deadband. Commands under ~4 ticks produce no motion at
# all and 4-8 work about 40% of the time, so a tighter tolerance can never be
# satisfied: on 2026-08-25 this loop sat 3 ticks short of target and spun 4849
# times over 123 s. 12 ticks is 0.3 mm, irrelevant to what this test measures.
ARRIVE_TOL = 12
MAX_ITERS = 400


def decode_load(raw):
    """Present_Load is SIGN-MAGNITUDE: bit 10 is direction, bits 0-9 the value.

    Reading it raw makes an idle motor look like ~1070 rather than ~45, which on
    2026-08-25 tripped this tool's own abort threshold on the first sample of
    every motor and produced a run in which nothing was ever pulled.
    """
    if raw is None:
        return None
    return -(raw & 0x3FF) if (raw & 0x400) else (raw & 0x3FF)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="shoggoth_mini/configs/default_hardware.yaml")
    ap.add_argument("--motors", default="1,2,3")
    ap.add_argument("--ticks", type=int, default=150,
                    help="how far to wind in (positive = tighten)")
    ap.add_argument("--step", type=int, default=15, help="ticks per increment")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--hold", type=float, default=1.0,
                    help="seconds to hold at full pull before releasing")
    ap.add_argument("--max-load", type=int, default=400,
                    help="abort this motor if the DECODED load magnitude exceeds "
                         "this (0-1023 scale; an idle motor reads ~45)")
    ap.add_argument("--csv", default="diagnostics/tendon_pull.csv")
    ap.add_argument("--plot", default="diagnostics/tendon_pull.png")
    args = ap.parse_args()

    motors = [m.strip() for m in args.motors.split(",") if m.strip()]
    if any(m not in MOTOR_NAMES for m in motors):
        raise SystemExit(f"unknown motor in {motors}; valid: {MOTOR_NAMES}")
    if args.ticks <= 0:
        raise SystemExit("--ticks must be positive: this test only winds IN")

    mc = MotorController(get_hardware_config(args.config))
    mc.connect()

    def rd(field, m):
        try:
            v = mc._motor_bus.read(field, m)
            return int(v.item() if hasattr(v, "item") else v[0])
        except Exception:
            return None

    start = mc.get_positions()
    print(f"\nstarting positions: {start}")
    print(f"winding each of {motors} in by {args.ticks} ticks "
          f"({args.ticks * TICKS_TO_MM:.1f} mm of cable), one at a time")
    for m in motors:
        if not (0 <= start[m] + args.ticks < PERIOD):
            print(f"  motor {m}: {start[m]} + {args.ticks} leaves 0..{PERIOD-1} — "
                  f"reduce --ticks")
            mc.disconnect()
            return 1
    print("\nWatch the SPOOL, not the reported number: the encoder is upstream of")
    print("the horn, so a slipping horn reports a perfect move regardless.")
    if input("Type 'go' to begin: ").strip().lower() != "go":
        print("aborted — nothing moved.")
        mc.disconnect()
        return 1

    rec = Recorder(args.csv, FIELDS)
    period = 1.0 / args.hz
    results = {}

    try:
        for m in motors:
            base = mc.get_position(m)
            print(f"\n--- motor {m}: {base} -> {base + args.ticks} ---")
            peak_load, loads, aborted = 0, [], False

            for phase, target in (("pull", base + args.ticks), ("release", base)):
                cur = mc.get_position(m)
                iters, last_print = 0, 0.0
                while abs(target - cur) > ARRIVE_TOL:
                    iters += 1
                    if iters > MAX_ITERS:
                        print(f"  {phase}: stopped {abs(target - cur)} ticks short "
                              f"after {MAX_ITERS} iterations")
                        break
                    cur += int(np.clip(target - cur, -args.step, args.step))
                    mc.set_position(m, cur)
                    time.sleep(period)
                    pres = rd("Present_Position", m)
                    ld_raw = rd("Present_Load", m)
                    ld = decode_load(ld_raw)
                    if pres is None:
                        continue
                    err = cur - pres
                    rec.log(motor=m, phase=phase, commanded=cur, present=pres,
                            error=err, load=abs(ld) if ld is not None else "",
                            load_raw=ld_raw if ld_raw is not None else "")
                    if time.time() - last_print > 0.3:
                        print(f"  {phase}: {pres} -> {target}   load {abs(ld) if ld is not None else '-':>4}   ",
                              end="\r", flush=True)
                        last_print = time.time()
                    if ld is not None and phase == "pull":
                        loads.append(abs(ld))
                        peak_load = max(peak_load, abs(ld))
                        if abs(ld) > args.max_load:
                            print(f"  load {abs(ld)} exceeded --max-load — releasing")
                            aborted = True
                            break
                    cur = pres
                if aborted:
                    cur = mc.get_position(m)
                    n = 0
                    while abs(base - cur) > ARRIVE_TOL and n < MAX_ITERS:
                        n += 1
                        cur += int(np.clip(base - cur, -args.step, args.step))
                        mc.set_position(m, cur)
                        time.sleep(period)
                        cur = mc.get_position(m)
                    break
                if phase == "pull":
                    t_end = time.time() + args.hold
                    while time.time() < t_end:
                        pres = rd("Present_Position", m)
                        ld_raw = rd("Present_Load", m)
                        ld = decode_load(ld_raw)
                        rec.log(motor=m, phase="hold", commanded=target,
                                present=pres, error=(target - pres) if pres else "",
                                load=abs(ld) if ld is not None else "",
                                load_raw=ld_raw if ld_raw is not None else "")
                        if ld is not None:
                            loads.append(abs(ld))
                            peak_load = max(peak_load, abs(ld))
                        time.sleep(period)

            final = mc.get_position(m)
            baseline = float(np.median(loads[:3])) if len(loads) >= 3 else 0.0
            results[m] = dict(peak=peak_load, baseline=baseline,
                              rise=peak_load - baseline,
                              returned=final - base, aborted=aborted)
            print(f"  peak load {peak_load}, rise above baseline "
                  f"{peak_load - baseline:.0f}, returned to {final} "
                  f"(offset {final - base:+d})")

    except KeyboardInterrupt:
        print("\nstopped — returning motors to start")
        for m in motors:
            try:
                cur = mc.get_position(m)
                n = 0
                while abs(start[m] - cur) > ARRIVE_TOL and n < MAX_ITERS:
                    n += 1
                    cur += int(np.clip(start[m] - cur, -args.step, args.step))
                    mc.set_position(m, cur)
                    time.sleep(period)
                    cur = mc.get_position(m)
            except Exception:
                pass
    finally:
        rec.close()
        end = mc.get_positions()
        print(f"\nfinal positions: {end}")
        drift = {m: end[m] - start[m] for m in MOTOR_NAMES}
        if any(abs(v) > 10 for v in drift.values()):
            print(f"  NOTE: not back where we started: {drift}")
        mc.disconnect()

    if not results:
        return 1

    print(f"\n=== TENDON PULL CHECK ===")
    print(f"{'motor':>6}{'baseline':>10}{'peak load':>11}{'rise':>8}   verdict")
    print("-" * 62)
    rises = {m: r["rise"] for m, r in results.items()}
    strongest = max(rises.values()) if rises else 0
    for m, r in results.items():
        if r["aborted"]:
            v = "hit the load limit — already under tension, or jammed"
        elif strongest > 0 and r["rise"] < 0.25 * strongest:
            v = "NOT PULLING — slack tendon, or horn slipping on the spline"
        elif r["rise"] < 30:
            v = "no load rise — nothing on the other end"
        else:
            v = "pulling"
        print(f"{m:>6}{r['baseline']:>10.0f}{r['peak']:>11}{r['rise']:>8.0f}   {v}")

    print("\ninterpretation:")
    if len(rises) > 1 and strongest > 0:
        weak = [m for m, v in rises.items() if v < 0.25 * strongest]
        if weak:
            print(f"    * motor(s) {', '.join(weak)} build far less load than the "
                  f"others for the same wind-in.")
            print(f"      If the spool visibly turned, the tendon is slack: tighten "
                  f"the knot, or retension.py --motors {weak[0]} --ticks 60 --apply")
            print(f"      If the spool did NOT turn, the horn is slipping on the "
                  f"spline and no amount of retensioning will help.")
        else:
            print("    * all tendons build comparable load — each is transmitting "
                  "force, so an unreachable direction is not a dead tendon.")
    print(f"    * loads are raw Present_Load counts; only the COMPARISON between "
          f"motors is meaningful here, not the absolute value.")

    _plot(args.csv, args.plot, results)
    print(f"csv  -> {args.csv}")
    return 0


def _plot(csv_path, out_path, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load(csv_path)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Tendon pull check — load developed per motor, wound in alone",
                 fontsize=12)

    colours = {"1": "tab:blue", "2": "tab:orange", "3": "tab:green"}
    for m in sorted(results):
        sel = [r for r in rows if r["motor"] == m]
        if len(sel) < 2:
            continue
        t = col(sel, "t")
        ax[0].plot(t - t[0], col(sel, "load"), lw=1.3, color=colours.get(m),
                   label=f"motor {m}")
    ax[0].set_xlabel("time within this motor's cycle (s)")
    ax[0].set_ylabel("Present_Load (raw counts)")
    ax[0].set_title("a tendon under tension makes load climb as it winds in",
                    fontsize=10)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    ms = sorted(results)
    ax[1].bar(range(len(ms)), [results[m]["rise"] for m in ms],
              color=[colours.get(m, "gray") for m in ms])
    ax[1].set_xticks(range(len(ms)))
    ax[1].set_xticklabels([f"motor {m}" for m in ms])
    ax[1].set_ylabel("load rise above baseline")
    ax[1].set_title("comparison is what matters — a bar near zero next to two "
                    "tall ones is the dead tendon", fontsize=10)
    ax[1].grid(alpha=.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130)
    print(f"plot -> {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
