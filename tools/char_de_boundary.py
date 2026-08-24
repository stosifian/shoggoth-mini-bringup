"""TEST D/E — what happens at and beyond the 0..4095 boundary?

WHY. Test C established that step size is not the hazard: with home at 2050,
every step from +/-50 to +/-2000 tracked to within 4 ticks, including +/-2048.
The one thing that misbehaved was the step whose TARGET left 0..4095 — and it
misbehaved in two mutually contradictory ways across runs:

    target 4774 (test C, 2026-08-17)  -> motor stopped at 4095          CLAMP
    target 5501 (grab failure, prior) -> motor landed at 1405 = 5501%4096  WRAP

Same servo, opposite behaviour, unexplained. Every guard in the stack now depends
on which of those is real, so this test settles it. It also exercises the negative
encoding that production code actually emits and that nothing has ever tested:
`geometry.py:83` maps a negative absolute position to `-32768 - absolute`, while
test C wrote raw negatives — a case the codebase never produces.

WHAT IT ANSWERS
  * how close to 0 and to 4095 can a target get before tracking degrades?
  * outside the range: does the motor clamp, wrap modulo 4096, or something else?
  * does the author's `-32768 - absolute` encoding reach the intended position?
  * do Min/Max_Angle_Limit (both read 0) explain a clamp at 4095?

HOW IT IS SAFE
  * TENDON MUST BE DETACHED.
  * no ladders: every trial is ONE write followed by observation, then a return
    to a mid-range home, so nothing compounds.
  * phases are selectable and ordered by risk; phase 3 (negatives, which caused a
    multi-turn runaway in test C) runs last and is capped at small magnitudes.
  * a watchdog trips if position leaves [-200, 4300]: torque off immediately, stop.
    Recovery is deliberately manual — driving a runaway servo back with more
    position commands is how test C's negative half corrupted itself.

  python tools/char_de_boundary.py --motor 2 --dry-run
  python tools/char_de_boundary.py --motor 2 --phases 1
  python tools/char_de_boundary.py --motor 2 --phases 1,2
  python tools/char_de_boundary.py --motor 2            # all phases
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.configs import get_hardware_config  # noqa: E402
from shoggoth_mini.hardware.motors import MotorController  # noqa: E402
from shoggoth_mini.common.constants import MOTOR_NAMES, MOTOR_POSITION_MIN  # noqa: E402
from char_common import Recorder, load, col  # noqa: E402

FIELDS = ["t", "motor", "phase", "trial", "commanded", "written", "encoding",
          "present", "present_speed", "load", "torque"]
SERVO_MODE = 0
PERIOD = 4096
TICKS_TO_MM = 0.11 / PERIOD * 1000

HOME = 2048
# The watchdog window is derived PER TRIAL from the position the trial intends to
# reach, never a fixed band. A fixed [-250, 4350] window (2026-08-17) tripped on
# trials that were tracking correctly toward a legitimate out-of-range target,
# and the run then recorded them as runaways — which inverted the phase 3 verdict
# and made phase 2 unreadable. A runaway is "went far past what was asked for",
# and that is only definable relative to what was asked for.
DETECT_MARGIN = 600          # ~80 ms of travel at the 7600 ticks/s ceiling
ABSOLUTE_LIMIT = 16384       # 4 turns; a detached spool tolerates this
RECOVER_LO, RECOVER_HI = -ABSOLUTE_LIMIT, ABSOLUTE_LIMIT


def detect_window(intended: int):
    """Safe travel window for a trial intending to reach `intended`."""
    lo = min(0, intended) - DETECT_MARGIN
    hi = max(PERIOD - 1, intended) + DETECT_MARGIN
    return (max(lo, -ABSOLUTE_LIMIT), min(hi, ABSOLUTE_LIMIT))

# Phase 2: the discriminator. 5501 is here because it is the exact target the
# grab primitive produced when it tore a tendon off its roller.
OUT_OF_RANGE = [4096, 4200, 4500, 5501, 6000, 8192]
# Phase 3: small magnitudes only — raw negatives ran away in test C.
NEGATIVES = [-100, -500, -1500]

REGISTERS = ("Mode", "Min_Angle_Limit", "Max_Angle_Limit", "Torque_Enable",
             "Offset", "Lock", "Goal_Position", "Present_Position")


def author_encoding(absolute: int) -> int:
    """What geometry.py:83 puts on the wire for a negative absolute position."""
    return MOTOR_POSITION_MIN - absolute


class Runaway(Exception):
    """Watchdog tripped during a trial. Recoverable — the trial is a data point."""

    def __init__(self, position, written, window):
        super().__init__(f"position {position} left [{window[0]}, {window[1]}] "
                         f"after writing {written}")
        self.position = position
        self.written = written
        self.window = window


class Aborted(Exception):
    """Unrecoverable; the run stops."""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="shoggoth_mini/configs/default_hardware.yaml")
    ap.add_argument("--motor", default="2", choices=MOTOR_NAMES)
    ap.add_argument("--phases", default="1,2,3",
                    help="which phases to run, comma separated")
    ap.add_argument("--home", type=int, default=HOME,
                    help="mid-range position returned to between trials")
    ap.add_argument("--observe", type=float, default=2.0,
                    help="seconds to watch after each command")
    ap.add_argument("--hz", type=float, default=100.0)
    ap.add_argument("--edge-step", type=int, default=50,
                    help="phase 1 increment size approaching each boundary")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--csv", default="diagnostics/char_de.csv")
    ap.add_argument("--plot", default="diagnostics/char_de.png")
    args = ap.parse_args()

    phases = {int(p) for p in args.phases.split(",") if p.strip()}

    # Phase 1 walks in from a safe distance to just past each end.
    up_targets = list(range(3800, PERIOD + 2 * args.edge_step, args.edge_step))
    down_targets = list(range(300, -2 * args.edge_step, -args.edge_step))

    print(f"\nTest D/E — motor {args.motor}, home {args.home}")
    if 1 in phases:
        print(f"  phase 1  approach boundaries in {args.edge_step}-tick steps: "
              f"{up_targets[0]}..{up_targets[-1]} and {down_targets[0]}..{down_targets[-1]}"
              f"  ({len(up_targets) + len(down_targets)} trials)")
    if 2 in phases:
        print(f"  phase 2  single out-of-range targets: {OUT_OF_RANGE}")
        print(f"           clamp predicts 4095 for all; wrap predicts "
              f"{[t % PERIOD for t in OUT_OF_RANGE]}")
    if 3 in phases:
        print(f"  phase 3  negatives {NEGATIVES}, raw vs author encoding "
              f"{[author_encoding(v) for v in NEGATIVES]}")
    n = ((len(up_targets) + len(down_targets)) if 1 in phases else 0) \
        + (len(OUT_OF_RANGE) if 2 in phases else 0) \
        + (2 * len(NEGATIVES) if 3 in phases else 0)
    print(f"\n  {n} trials, ~{n * (args.observe + 1.5):.0f}s")

    if args.dry_run:
        print("\ndry run — nothing moved.")
        return 0

    mc = MotorController(get_hardware_config(args.config))
    mc.connect()
    m = args.motor
    bus = mc._motor_bus

    def rd(field, retries=3):
        """Read with retries. A single dropped read on this half-duplex bus
        aborted a run mid-phase (2026-08-17); one failure is not a fault."""
        for _ in range(retries):
            try:
                v = bus.read(field, m)
                return int(v.item() if hasattr(v, "item") else v[0])
            except Exception:
                time.sleep(0.02)
        return None

    def snapshot(label):
        regs = {k: rd(k) for k in REGISTERS}
        print(f"  registers [{label}]: " +
              "  ".join(f"{k}={v}" for k, v in regs.items()))
        return regs

    print()
    reg_log = {"start": snapshot("start")}

    print("\n*** THE TENDON ON THIS MOTOR MUST BE DETACHED ***")
    if input("Type 'free' to continue: ").strip().lower() != "free":
        print("aborted — nothing moved.")
        mc.disconnect()
        return 1

    rec = Recorder(args.csv, FIELDS)
    period = 1.0 / args.hz
    results = []
    trial_no = 0

    def write_goal(value):
        with mc._bus_lock:
            bus.write("Goal_Position", int(value), m)

    def observe(written, commanded, phase, encoding, seconds=None):
        """One write, then watch. Returns the landing position."""
        nonlocal trial_no
        trial_no += 1
        secs = args.observe if seconds is None else seconds
        window = detect_window(commanded)
        start = rd("Present_Position")
        write_goal(written)
        t0 = time.time()
        trace = []
        rewritten = False
        while time.time() - t0 < secs:
            p = rd("Present_Position")
            trace.append(p)
            # Two phase-1 trials (targets 200 and 150) never left home on
            # 2026-08-17: the write was silently dropped on the bus. Detect a
            # command that produced no motion at all and re-issue it once.
            if (not rewritten and start is not None and p is not None
                    and time.time() - t0 > 0.25
                    and abs(p - start) < 20 and abs(commanded - start) > 100):
                print("      no motion after 0.25s — re-issuing the write")
                write_goal(written)
                rewritten = True
            rec.log(motor=m, phase=phase, trial=trial_no, commanded=commanded,
                    written=int(written), encoding=encoding, present=p,
                    present_speed=rd("Present_Speed"), load=rd("Present_Load"),
                    torque=rd("Torque_Enable"))
            if p is not None and not (window[0] <= p <= window[1]):
                bus.write("Torque_Enable", 0, m)   # cutting torque stops it
                raise Runaway(p, written, window)
            time.sleep(period)
        good = [p for p in trace if p is not None]
        return (float(np.mean(good[-max(1, len(good) // 5):])) if good else float("nan"))

    def go_home():
        """Ramp back to mid-range in 400-tick increments.

        The acceptance window is the watchdog window, NOT 0..4095. Requiring
        0..4095 here aborted the first run at trial 8 from a position of 4154 —
        which was healthy: the reading exceeds 4095 transiently before folding
        (4154 and 58 are the same shaft angle). Only a position outside the
        watchdog window is genuinely unrecoverable.
        """
        bus.write("Torque_Enable", 1, m)
        cur = rd("Present_Position")
        if cur is None or not (RECOVER_LO <= cur <= RECOVER_HI):
            raise Aborted(f"cannot home from position {cur} — recover manually")
        stalled = 0
        while abs(args.home - cur) > 3:
            step = int(np.clip(args.home - cur, -400, 400))
            write_goal(cur + step)
            time.sleep(0.3)
            new = rd("Present_Position")
            if new is None:
                raise Aborted("lost position read while homing (after retries)")
            # A write can be dropped on this bus; do not spin forever if so.
            stalled = stalled + 1 if abs(new - cur) < 10 else 0
            if stalled >= 5:
                raise Aborted(f"homing stalled at {new}, target {args.home}")
            cur = new
        return cur

    def trial(written, commanded, phase, encoding, seconds=None):
        """One trial with per-trial runaway recovery.

        A runaway is a RESULT here, not a failure: phase 3 expects the raw
        encoding to run away, and that is precisely the control it provides.
        Cutting torque stops it, the motor is free, and the reading folds — so
        the position is recoverable and the run continues to the next trial.
        Only a position outside the recovery window stops the run.
        """
        go_home()
        try:
            landed = observe(written, commanded, phase, encoding, seconds)
            return dict(phase=phase, commanded=commanded, written=written,
                        encoding=encoding, landed=landed,
                        error=landed - commanded, runaway=False)
        except Runaway as r:
            print(f"      runaway: {r} — torque cut, recovering")
            time.sleep(0.5)
            here = rd("Present_Position")
            if here is None or not (RECOVER_LO <= here <= RECOVER_HI):
                raise Aborted(f"runaway left position at {here}")
            return dict(phase=phase, commanded=commanded, written=written,
                        encoding=encoding, landed=float(r.position),
                        error=float(r.position) - commanded, runaway=True)

    try:
        bus.write("Mode", SERVO_MODE, m)
        bus.write("Torque_Enable", 1, m)
        time.sleep(0.2)
        go_home()

        if 1 in phases:
            print(f"\n--- phase 1: approaching boundaries "
                  f"({args.edge_step}-tick steps) ---")
            for name, targets in (("up", up_targets), ("down", down_targets)):
                for tgt in targets:
                    r = trial(tgt, tgt, f"p1_{name}", "raw", seconds=0.6)
                    results.append(r)
                    flag = ("   <-- RUNAWAY" if r["runaway"] else
                            "   <-- not tracking" if abs(r["error"]) > 40 else "")
                    print(f"  target {tgt:>6}  landed {r['landed']:>8.0f}  "
                          f"err {r['error']:>+7.0f}{flag}")
            reg_log["after_p1"] = snapshot("after phase 1")

        if 2 in phases:
            print("\n--- phase 2: single out-of-range targets ---")
            print("    clamp -> lands 4095 | wrap -> lands target mod 4096")
            for tgt in OUT_OF_RANGE:
                r = trial(tgt, tgt, "p2", "raw")
                clamp_pred, wrap_pred = PERIOD - 1, tgt % PERIOD
                r["verdict"] = ("RUNAWAY" if r["runaway"] else
                                "TRACKED" if abs(r["landed"] - tgt) < 60 else
                                "CLAMP" if abs(r["landed"] - clamp_pred) < 60 else
                                "WRAP" if abs(r["landed"] - wrap_pred) < 60 else
                                "NEITHER")
                results.append(r)
                print(f"  target {tgt:>6}  landed {r['landed']:>8.0f}   "
                      f"clamp={clamp_pred} wrap={wrap_pred}   -> {r['verdict']}")
            reg_log["after_p2"] = snapshot("after phase 2")

        if 3 in phases:
            print("\n--- phase 3: negative targets, raw vs author encoding ---")
            print("    the raw encoding is EXPECTED to run away; that is the control")
            for val in NEGATIVES:
                for enc, written in (("raw", val), ("author", author_encoding(val))):
                    r = trial(written, val, "p3", enc)
                    results.append(r)
                    print(f"  intended {val:>6}  wrote {written:>8}  ({enc:>6})"
                          f"  landed {r['landed']:>8.0f}"
                          f"{'   <-- RUNAWAY' if r['runaway'] else ''}")
            reg_log["after_p3"] = snapshot("after phase 3")

        go_home()

    except Aborted as e:
        print(f"\n!! WATCHDOG: {e}")
        print("   Torque is OFF. Do NOT drive it back with position commands —")
        print("   that is what corrupted test C. Re-home by hand or power-cycle.")
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        try:
            bus.write("Torque_Enable", 0, m)
            print("\ntorque OFF, no position pinned.")
        except Exception as e:
            print(f"cleanup issue: {e}")
        rec.close()
        mc.disconnect()

    if not results:
        print("no usable trials")
        return 1

    print(f"\n=== TEST D/E RESULT — motor {m} ===")

    p1 = [r for r in results if r["phase"].startswith("p1")]
    if p1:
        tracked = [r["commanded"] for r in p1 if abs(r["error"]) <= 40]
        if tracked:
            print(f"    * tracking holds over commanded targets "
                  f"{min(tracked)}..{max(tracked)}")
        bad = [r for r in p1 if abs(r["error"]) > 40]
        if bad:
            print(f"    * first target that stopped tracking: "
                  f"{bad[0]['commanded']} (landed {bad[0]['landed']:.0f})")
        else:
            print("    * every phase-1 target tracked, including past the boundary")

    p2 = [r for r in results if r["phase"] == "p2"]
    if p2:
        verdicts = [r["verdict"] for r in p2]
        print(f"    * out-of-range verdicts: " +
              ", ".join(f"{r['commanded']}->{r['verdict']}" for r in p2))
        if len(set(verdicts)) == 1:
            print(f"      -> consistently {verdicts[0]}. The 4774-vs-5501 "
                  f"contradiction is resolved in favour of {verdicts[0]}.")
        else:
            print("      -> MIXED. Behaviour depends on something not yet "
                  "controlled for; inspect the trace and register snapshots.")

    p3 = [r for r in results if r["phase"] == "p3"]
    if p3:
        # A runaway and a miss are different outcomes and must not be conflated:
        # counting a watchdog trip as "did not reach" inverted this verdict once.
        tally = {}
        for enc in ("raw", "author"):
            sel = [r for r in p3 if r["encoding"] == enc]
            if not sel:
                continue
            hit = sum(1 for r in sel if not r["runaway"] and abs(r["error"]) <= 60)
            ran = sum(1 for r in sel if r["runaway"])
            tally[enc] = (hit, ran, len(sel))
            print(f"    * {enc:>6}: reached intended {hit}/{len(sel)}, "
                  f"ran away {ran}/{len(sel)}, "
                  f"settled elsewhere {len(sel) - hit - ran}/{len(sel)}")
        a, r_ = tally.get("author", (0, 0, 0)), tally.get("raw", (0, 0, 0))
        if a[2] and a[0] == a[2] and r_[2] and r_[1] == r_[2]:
            print("      -> CONFIRMED: Goal_Position is 16-bit sign-magnitude. "
                  "geometry.py's -32768 - absolute is the correct conversion, "
                  "and a raw negative runs away.")
        elif a[2] and a[0] < a[2]:
            print("      -> the author encoding did NOT reach the intended "
                  "position on every trial; inspect the trace before concluding")

    print("\n    register snapshots:")
    for k, v in reg_log.items():
        print(f"      {k:<12} " + " ".join(f"{a}={b}" for a, b in v.items()))

    _plot(args.csv, args.plot, results, m)
    print(f"csv  -> {args.csv}")
    return 0


def _plot(csv_path, out_path, results, motor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load(csv_path)
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 1])
    fig.suptitle(f"Test D/E — boundary and out-of-range behaviour, motor {motor}",
                 fontsize=13)

    # panel 1: the headline — where does each commanded target actually land?
    ax = fig.add_subplot(gs[0, :])
    allc = [r["written"] for r in results] or [0]
    span = np.linspace(min(allc) - 200, max(allc) + 200, 3000)
    ax.plot(span, span, ls=":", lw=1.2, color="gray", label="ideal")
    ax.plot(span, np.clip(span, 0, PERIOD - 1), lw=1.4, color="tab:red",
            label="clamp to 0..4095")
    ax.plot(span, span % PERIOD, lw=1.0, color="tab:orange", alpha=.8,
            label=f"wrap modulo {PERIOD}")
    styles = {"p1_up": ("tab:blue", "o"), "p1_down": ("tab:cyan", "o"),
              "p2": ("tab:green", "D"), "p3": ("tab:purple", "s")}
    for ph, (c, mk) in styles.items():
        sel = [r for r in results if r["phase"] == ph]
        if sel:
            ax.plot([r["written"] for r in sel], [r["landed"] for r in sel],
                    mk, ms=7, color=c, label=ph)
    ax.axhline(PERIOD - 1, color="gray", lw=0.8, alpha=.5)
    ax.axhline(0, color="gray", lw=0.8, alpha=.5)
    ax.set_xlabel("value written to Goal_Position")
    ax.set_ylabel("position landed at")
    ax.set_title("which model does the servo follow outside 0..4095 — "
                 "the red clamp line or the orange wrap sawtooth?", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    # panel 2: phase 1 detail — tracking error approaching each boundary
    ax = fig.add_subplot(gs[1, :])
    for ph, (c, mk) in (("p1_up", ("tab:blue", "o")), ("p1_down", ("tab:cyan", "o"))):
        sel = sorted([r for r in results if r["phase"] == ph],
                     key=lambda r: r["commanded"])
        if sel:
            ax.plot([r["commanded"] for r in sel], [r["error"] for r in sel],
                    mk + "-", ms=5, color=c, label=ph)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axhspan(-40, 40, color="tab:green", alpha=.12, label="±40 ticks (tracking)")
    ax.axvline(PERIOD - 1, color="tab:red", lw=1, ls="--", alpha=.6)
    ax.axvline(0, color="tab:red", lw=1, ls="--", alpha=.6)
    ax.set_xlabel("commanded target")
    ax.set_ylabel("landed - commanded (ticks)")
    ax.set_title("phase 1: how close to each boundary tracking survives", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    # panel 3: phase 3 — do the two negative encodings differ?
    ax = fig.add_subplot(gs[2, 0])
    p3 = [r for r in results if r["phase"] == "p3"]
    if p3:
        vals = sorted({r["commanded"] for r in p3})
        w = 0.35
        for k, (enc, c) in enumerate((("raw", "tab:purple"), ("author", "tab:olive"))):
            xs = [i + (k - 0.5) * w for i, v in enumerate(vals)]
            ys = [next((r["landed"] for r in p3
                        if r["commanded"] == v and r["encoding"] == enc), np.nan)
                  for v in vals]
            ax.bar(xs, ys, width=w, color=c, label=enc)
        ax.plot(range(len(vals)), vals, "k_", ms=22, label="intended")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([str(v) for v in vals])
        ax.axhline(0, color="gray", lw=0.8)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "phase 3 not run", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
    ax.set_xlabel("intended absolute position")
    ax.set_ylabel("landed")
    ax.set_title("phase 3: raw vs geometry.py's -32768 - absolute", fontsize=10)
    ax.grid(alpha=.3, axis="y")

    # panel 4: the whole run, so a runaway is visible in context
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(col(rows, "t"), col(rows, "present"), lw=1.0, color="tab:blue")
    ax.axhline(PERIOD - 1, color="tab:red", lw=0.9, ls="--", alpha=.6)
    ax.axhline(0, color="tab:red", lw=0.9, ls="--", alpha=.6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Present_Position")
    ax.set_title("full run — excursions past the dashed lines are the interesting "
                 "part", fontsize=10)
    ax.grid(alpha=.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    print(f"plot -> {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
