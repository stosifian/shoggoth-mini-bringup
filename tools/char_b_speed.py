"""TEST B — characterise Goal_Speed: what does the number actually mean?

WHY. Test A2 asked for magnitude 250 in both directions and measured 230 ticks/s
forward but 6912 ticks/s reverse — a 30x asymmetry, not the ~1x the code implies.
The encoding used is the author's, from `calibrate`:

    forward: magnitude
    reverse: -(1024 - magnitude)

So the tool you wind every tendon with currently changes speed by roughly 30x when
you press spacebar, not just direction. That is a live hazard and it is unmeasured.
This measures it.

WHAT IT ANSWERS
  * what real speed (ticks/s, mm/s of cable) does each commanded magnitude give?
  * are the two directions symmetric at the same magnitude?
  * is the relationship linear, and where does it saturate?
  * is `-(1024 - magnitude)` the right reverse encoding, or is plain `-magnitude`?

HOW IT IS SAFE
  * wheel mode only: it commands SPEED, never a position, so the modulo-4096
    shortest-path behaviour that has caused every failure this week cannot occur.
  * TENDON MUST BE DETACHED — continuous rotation spools wire without limit.
  * short bursts (default 3 s), Goal_Speed zeroed between every step, and the
    ladder starts at the SMALLEST magnitude so the first surprise is the mildest.
  * Ctrl+C stops and zeroes speed.

  python tools/char_b_speed.py --motor 2
  python tools/char_b_speed.py --motor 2 --magnitudes 100,200,400 --burst 2
  python tools/char_b_speed.py --motor 2 --encoding plain   # test -magnitude instead
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

FIELDS = ["t", "motor", "phase", "magnitude", "direction", "encoding",
          "goal_speed_written", "present", "present_speed", "load"]
WHEEL_MODE, SERVO_MODE = 1, 0
TICKS_TO_MM = 0.11 / 4096 * 1000


def encode(direction: int, magnitude: int, scheme: str) -> int:
    """Two candidate reverse encodings, so the test can distinguish them."""
    magnitude = int(min(abs(magnitude), 1023))
    if direction >= 0:
        return magnitude
    if scheme == "author":
        return -(1024 - magnitude)   # what calibrate.py does
    return -magnitude                # the obvious alternative


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="shoggoth_mini/configs/default_hardware.yaml")
    ap.add_argument("--motor", default="2", choices=MOTOR_NAMES)
    ap.add_argument("--magnitudes", default="100,200,400,800")
    ap.add_argument("--encoding", choices=["author", "plain", "both"], default="both")
    ap.add_argument("--burst", type=float, default=3.0, help="seconds per step")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--csv", default="diagnostics/char_b.csv")
    ap.add_argument("--plot", default="diagnostics/char_b.png")
    args = ap.parse_args()

    mags = sorted(int(x) for x in args.magnitudes.split(",") if x.strip())
    schemes = ["author", "plain"] if args.encoding == "both" else [args.encoding]

    mc = MotorController(get_hardware_config(args.config))
    mc.connect()
    m = args.motor
    bus = mc._motor_bus

    def rd(f):
        try:
            v = bus.read(f, m)
            return int(v.item() if hasattr(v, "item") else v[0])
        except Exception:
            return None

    print(f"\nmotor {m}: magnitudes {mags}, encodings {schemes}, "
          f"{args.burst:.0f}s per step")
    print(f"total motion time ~{len(mags)*len(schemes)*2*args.burst:.0f}s")
    print("\n*** TENDON ON THIS MOTOR MUST BE DETACHED ***")
    print("Speed commands only — no position targets — but rotation is unbounded.")
    if input("Type 'free' to continue: ").strip().lower() != "free":
        print("aborted — nothing moved.")
        mc.disconnect()
        return 1

    rec = Recorder(args.csv, FIELDS)
    period = 1.0 / args.hz
    results = []

    def burst(direction, magnitude, scheme):
        gs = encode(direction, magnitude, scheme)
        label = f"{scheme}/{'fwd' if direction > 0 else 'rev'}/{magnitude}"
        # Mode first, torque after: the Mode write clears Torque_Enable.
        bus.write("Mode", WHEEL_MODE, m)
        bus.write("Goal_Speed", 0, m)
        bus.write("Torque_Enable", 1, m)
        samples = []
        t0 = time.time()
        bus.write("Goal_Speed", gs, m)
        while time.time() - t0 < args.burst:
            p = rd("Present_Position")
            samples.append((time.time() - t0, p))
            rec.log(motor=m, phase=label, magnitude=magnitude,
                    direction=direction, encoding=scheme, goal_speed_written=gs,
                    present=p, present_speed=rd("Present_Speed"), load=rd("Present_Load"))
            time.sleep(period)
        bus.write("Goal_Speed", 0, m)
        time.sleep(0.5)

        ts = np.array([s[0] for s in samples], dtype=float)
        ps = np.array([np.nan if s[1] is None else s[1] for s in samples], dtype=float)
        unw = unwrap_ticks(ps)
        ok = ~np.isnan(unw)
        if ok.sum() < 5:
            return None
        # least-squares slope is robust to the odd dropped read
        slope = np.polyfit(ts[ok], unw[ok], 1)[0]
        travelled = unw[ok][-1] - unw[ok][0]
        return dict(scheme=scheme, direction=direction, magnitude=magnitude,
                    written=gs, ticks_per_s=slope, travelled=travelled,
                    turns=travelled / 4096.0)

    try:
        for scheme in schemes:
            for mag in mags:                     # smallest first, deliberately
                for direction in (+1, -1):
                    r = burst(direction, mag, scheme)
                    if r:
                        results.append(r)
                        print(f"  {scheme:>6} {'fwd' if direction>0 else 'rev'} "
                              f"mag {mag:>4} -> wrote {r['written']:>6}  "
                              f"{abs(r['ticks_per_s']):>7.0f} ticks/s  "
                              f"({abs(r['ticks_per_s'])*TICKS_TO_MM:>6.1f} mm/s)  "
                              f"{r['turns']:+.2f} turns")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        try:
            bus.write("Goal_Speed", 0, m)
            bus.write("Mode", SERVO_MODE, m)
            print("\nGoal_Speed 0, Mode restored to servo. Torque left as-is.")
        except Exception as e:
            print(f"cleanup issue: {e}")
        rec.close()
        mc.disconnect()

    if not results:
        print("no usable bursts")
        return 1

    print(f"\n=== TEST B RESULT — motor {m} ===")
    print(f"{'encoding':>8}{'dir':>5}{'mag':>6}{'written':>9}{'ticks/s':>10}{'mm/s':>8}")
    print("-" * 46)
    for r in results:
        print(f"{r['scheme']:>8}{'fwd' if r['direction']>0 else 'rev':>5}"
              f"{r['magnitude']:>6}{r['written']:>9}"
              f"{abs(r['ticks_per_s']):>10.0f}{abs(r['ticks_per_s'])*TICKS_TO_MM:>8.1f}")

    print("\ninterpretation:")
    for scheme in schemes:
        fwd = {r["magnitude"]: abs(r["ticks_per_s"])
               for r in results if r["scheme"] == scheme and r["direction"] > 0}
        rev = {r["magnitude"]: abs(r["ticks_per_s"])
               for r in results if r["scheme"] == scheme and r["direction"] < 0}
        common = sorted(set(fwd) & set(rev))
        if not common:
            continue
        ratios = [rev[k] / fwd[k] for k in common if fwd[k] > 1]
        if ratios:
            worst = max(ratios)
            print(f"    {scheme}: reverse/forward speed ratio "
                  f"{min(ratios):.2f}..{worst:.2f}")
            if worst > 1.5 or min(ratios) < 0.67:
                print(f"      -> ASYMMETRIC. This encoding does not preserve speed "
                      f"across direction.")
            else:
                print(f"      -> symmetric within ~50%: this is the sane encoding.")

    _plot(args.csv, args.plot, results, m)
    print(f"csv  -> {args.csv}")
    return 0


def _plot(csv_path, out_path, results, motor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load(csv_path)
    fig, ax = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle(f"Test B — Goal_Speed characterisation, motor {motor}", fontsize=12)

    # panel 1: commanded magnitude -> measured speed, per encoding+direction
    styles = {("author", 1): ("tab:blue", "o-"), ("author", -1): ("tab:red", "o--"),
              ("plain", 1): ("tab:green", "s-"), ("plain", -1): ("tab:orange", "s--")}
    for (scheme, direction), (colr, st) in styles.items():
        pts = sorted([(r["magnitude"], abs(r["ticks_per_s"])) for r in results
                      if r["scheme"] == scheme and r["direction"] == direction])
        if not pts:
            continue
        ax[0].plot([p[0] for p in pts], [p[1] for p in pts], st, color=colr,
                   label=f"{scheme} {'forward' if direction > 0 else 'reverse'}")
    ax[0].set_xlabel("commanded magnitude")
    ax[0].set_ylabel("measured ticks/s")
    ax[0].set_title("commanded magnitude vs measured speed — matching solid/dashed "
                    "pairs mean the encoding is direction-symmetric", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)
    sec = ax[0].secondary_yaxis('right', functions=(lambda t: t * TICKS_TO_MM,
                                                    lambda s: s / TICKS_TO_MM))
    sec.set_ylabel("mm/s of cable")

    # panel 2: the raw traces, so a stall or saturation is visible
    t = col(rows, "t")
    pos = unwrap_ticks(col(rows, "present"))
    ax[1].plot(t, pos, lw=1.2, color="tab:purple")
    phases = [r.get("phase", "") for r in rows]
    last = None
    for i, ph in enumerate(phases):
        if ph != last:
            ax[1].axvline(t[i], color="gray", lw=0.6, alpha=.4)
            last = ph
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("unwrapped ticks")
    ax[1].set_title("unwrapped position through the whole run; grey lines separate "
                    "steps (slope = speed)", fontsize=10)
    ax[1].grid(alpha=.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    print(f"plot -> {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
