"""BUS WRITE RELIABILITY — how often does a Goal_Position write silently vanish?

WHY. During test D/E (2026-08-17) two consecutive trials commanded targets of 200
and 150 from a park of 2048, and the motor did not move at all — it sat at 2049
for the whole observation window, then the next trial worked normally. No
exception was raised. The write was accepted by the API and did nothing.

That matters well beyond the characterisation series. A dropped write leaves the
PREVIOUS target standing, so the motor holds a stale position while the caller
believes it commanded a new one. In a loop that streams targets this is invisible;
the next frame overwrites it. But a primitive that issues one target and sleeps
would simply not move, and the frame AFTER a dropped write can present as a large
sudden correction — which is a candidate explanation for the abrupt startup
motions seen during `orchestrate`.

WHAT IT ANSWERS
  * what fraction of writes fail to take effect, per motor?
  * do they raise, or fail silently?
  * does `Goal_Position` read back as what was written?
  * does driving three motors (the real code path) drop more than driving one?
  * what loop rate does a write+read cycle actually sustain on this bus?

HOW IT IS SAFE
  * amplitude is +/-20 ticks around the park position — under 1 mm of cable.
  * every target is clamped to 0..4095.
  * safe with tendons threaded but untied: the motion is tiny and symmetric
    about the park point, so no net cable is paid out over the run.

  python tools/char_bus_writes.py --dry-run
  python tools/char_bus_writes.py --trials 2000
  python tools/char_bus_writes.py --trials 2000 --mode single
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.configs import get_hardware_config  # noqa: E402
from shoggoth_mini.hardware.motors import MotorController  # noqa: E402
from shoggoth_mini.common.constants import MOTOR_NAMES  # noqa: E402
from char_common import Recorder, load, col  # noqa: E402

FIELDS = ["t", "trial", "mode", "motor", "target", "goal_readback", "present",
          "prev_present", "moved", "write_error", "read_error", "period"]
PERIOD = 4096
TICKS_TO_MM = 0.11 / PERIOD * 1000

PARK = 2048
AMPLITUDE = 20          # +/- ticks; 20 ticks is 0.54 mm of cable
NO_MOTION_TICKS = 3     # below this counts as "did not move"
MIN_EXPECTED = 8        # only judge motion when the command asked for this much


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="shoggoth_mini/configs/default_hardware.yaml")
    ap.add_argument("--motors", default="1,2,3")
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--mode", choices=["single", "all", "both"], default="both",
                    help="single: one motor per write. all: set_positions for "
                         "every motor, which is what the control loops do.")
    ap.add_argument("--park", type=int, default=PARK)
    ap.add_argument("--amplitude", type=int, default=AMPLITUDE)
    ap.add_argument("--settle", type=float, default=0.05,
                    help="seconds between write and the position read")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--csv", default="diagnostics/char_bus.csv")
    ap.add_argument("--plot", default="diagnostics/char_bus.png")
    args = ap.parse_args()

    motors = [m.strip() for m in args.motors.split(",") if m.strip()]
    if any(m not in MOTOR_NAMES for m in motors):
        raise SystemExit(f"unknown motor in {motors}; valid: {MOTOR_NAMES}")
    modes = ["single", "all"] if args.mode == "both" else [args.mode]

    lo, hi = args.park - args.amplitude, args.park + args.amplitude
    if not (0 <= lo and hi <= PERIOD - 1):
        raise SystemExit(f"park {args.park} +/- {args.amplitude} leaves 0..{PERIOD-1}")

    print(f"\nbus write reliability — motors {motors}, modes {modes}")
    print(f"  {args.trials} trials per mode, targets in {lo}..{hi} "
          f"(+/-{args.amplitude * TICKS_TO_MM:.2f} mm of cable)")
    print(f"  safe with tendons threaded but untied: motion is symmetric about "
          f"the park point, so no net cable is paid out")

    if args.dry_run:
        print("\ndry run — nothing moved.")
        return 0

    rng = random.Random(args.seed)
    mc = MotorController(get_hardware_config(args.config))
    mc.connect()
    bus = mc._motor_bus

    def rd(field, motor):
        try:
            v = bus.read(field, motor)
            return int(v.item() if hasattr(v, "item") else v[0]), None
        except Exception as e:
            return None, type(e).__name__

    print("\nparking before the run so every target is a small move from rest")
    for m in motors:
        cur, _ = rd("Present_Position", m)
        if cur is None:
            print(f"  motor {m}: cannot read position — skipping")
            continue
        if not (0 <= cur <= PERIOD - 1):
            print(f"  motor {m}: at {cur}, outside 0..{PERIOD-1}. "
                  f"Park it first with tools/park_motors.py")
            mc.disconnect()
            return 1
        while abs(args.park - cur) > 3:
            cur += int(np.clip(args.park - cur, -3, 3))
            mc.set_position(m, cur)
            time.sleep(0.02)
        print(f"  motor {m}: parked at {cur}")

    rec = Recorder(args.csv, FIELDS)
    prev_present = {m: rd("Present_Position", m)[0] for m in motors}
    prev_target = {m: args.park for m in motors}
    stats = {(mode, m): dict(writes=0, write_err=0, read_err=0, mismatch=0,
                             no_motion=0, judged=0)
             for mode in modes for m in motors}

    try:
        for mode in modes:
            print(f"\n--- mode: {mode} ---")
            t_last = time.time()
            for i in range(args.trials):
                targets = {m: rng.randint(lo, hi) for m in motors}

                write_err = {}
                if mode == "all":
                    try:
                        mc.set_positions(targets)
                    except Exception as e:
                        write_err = {m: type(e).__name__ for m in motors}
                else:
                    for m in motors:
                        try:
                            mc.set_position(m, targets[m])
                        except Exception as e:
                            write_err[m] = type(e).__name__

                readback = {m: rd("Goal_Position", m)[0] for m in motors}
                time.sleep(args.settle)

                now = time.time()
                period = now - t_last
                t_last = now

                for m in motors:
                    present, read_err = rd("Present_Position", m)
                    s = stats[(mode, m)]
                    s["writes"] += 1
                    if write_err.get(m):
                        s["write_err"] += 1
                    if read_err:
                        s["read_err"] += 1
                    if readback[m] is not None and readback[m] != targets[m]:
                        s["mismatch"] += 1

                    # Only judge motion when the command actually asked for some.
                    # This must be measured against the motor's ACTUAL previous
                    # position, not the previous target: measuring it
                    # target-to-target (2026-08-18) counted "already there" as a
                    # failure and invented a ~1% drop rate on two clean motors.
                    asked = (abs(targets[m] - prev_present[m])
                             if prev_present[m] is not None else 0)
                    moved = (abs(present - prev_present[m])
                             if present is not None and prev_present[m] is not None
                             else None)
                    if asked >= MIN_EXPECTED and moved is not None:
                        s["judged"] += 1
                        if moved < NO_MOTION_TICKS:
                            s["no_motion"] += 1

                    rec.log(trial=i, mode=mode, motor=m, target=targets[m],
                            goal_readback=readback[m], present=present,
                            prev_present=prev_present[m], moved=moved,
                            write_error=write_err.get(m, ""), read_error=read_err or "",
                            period=f"{period:.5f}")
                    prev_present[m] = present
                    prev_target[m] = targets[m]

                if i % 50 == 0:
                    print(f"  trial {i}/{args.trials}   ", end="\r", flush=True)
            print(f"  {args.trials} trials done" + " " * 20)

    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        rec.close()
        mc.disconnect()

    print(f"\n=== BUS WRITE RELIABILITY ===")
    print(f"{'mode':>8}{'motor':>7}{'writes':>8}{'wr err':>8}{'rd err':>8}"
          f"{'readback≠':>11}{'no motion':>11}{'rate':>9}")
    print("-" * 70)
    for (mode, m), s in stats.items():
        if not s["writes"]:
            continue
        pct = (100.0 * s["no_motion"] / s["judged"]) if s["judged"] else 0.0
        print(f"{mode:>8}{m:>7}{s['writes']:>8}{s['write_err']:>8}"
              f"{s['read_err']:>8}{s['mismatch']:>11}"
              f"{s['no_motion']:>7}/{s['judged']:<4}{pct:>8.2f}%")

    total_no_motion = sum(s["no_motion"] for s in stats.values())
    total_judged = sum(s["judged"] for s in stats.values())
    total_err = sum(s["write_err"] + s["read_err"] for s in stats.values())
    print("\ninterpretation:")
    if total_judged:
        rate = 100.0 * total_no_motion / total_judged
        print(f"    * {total_no_motion}/{total_judged} commands produced no motion "
              f"({rate:.2f}%)")
        if total_no_motion == 0:
            print("      -> no silent drops in this run. The D/E observation was "
                  "either rarer than this sample, or specific to large moves.")
        elif rate < 0.5:
            print("      -> rare but real. At 200 Hz that is still several per "
                  "minute; a primitive that issues one target can miss.")
        else:
            print("      -> frequent enough to explain missed primitives and "
                  "apparent sudden corrections.")
    if total_err:
        print(f"    * {total_err} raised errors — these are the LOUD failures, "
              f"already visible to callers")
    else:
        print("    * no exceptions raised: any failure here is silent")

    rows = load(args.csv)
    per = col(rows, "period")
    per = per[~np.isnan(per) & (per > 0)]
    if len(per):
        # The measured period includes --settle, which is our own sleep and says
        # nothing about the bus. Reporting the raw figure as a bus rate (2026-08-18)
        # made a 130 Hz bus look like 17 Hz.
        bus_ms = np.median(per) * 1000 - args.settle * 1000
        n_tx = len(motors) * 3          # 1 write + Goal readback + Present read
        loop_tx = len(motors) * 2       # what a control loop actually does
        print(f"    * cycle {np.median(per)*1000:.1f} ms, of which "
              f"{args.settle*1000:.0f} ms is this tool's settle sleep")
        print(f"      -> {bus_ms:.1f} ms of bus work for {n_tx} transactions "
              f"= {bus_ms/n_tx:.2f} ms each")
        print(f"      -> a {loop_tx}-transaction control loop costs "
              f"~{bus_ms/n_tx*loop_tx:.1f} ms = {1000/(bus_ms/n_tx*loop_tx):.0f} Hz "
              f"(configured for 200 Hz)")

    _plot(args.csv, args.plot, stats, motors)
    print(f"csv  -> {args.csv}")
    return 0


def _plot(csv_path, out_path, stats, motors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load(csv_path)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Bus write reliability", fontsize=13)

    # panel 1: readback error — did Goal_Position take the value we wrote?
    tgt, rb = col(rows, "target"), col(rows, "goal_readback")
    diff = rb - tgt
    finite = ~np.isnan(diff)
    ax[0][0].hist(diff[finite], bins=61, color="tab:blue")
    ax[0][0].set_yscale("log")
    ax[0][0].set_xlabel("Goal_Position readback - target (ticks)")
    ax[0][0].set_ylabel("count (log)")
    ax[0][0].set_title(f"readback mismatch — {int((diff[finite] != 0).sum())} of "
                       f"{int(finite.sum())} writes did not read back",
                       fontsize=10)
    ax[0][0].grid(alpha=.3)

    # panel 2: motion produced, with the no-motion band marked
    moved = col(rows, "moved")
    mfin = moved[~np.isnan(moved)]
    ax[0][1].hist(mfin, bins=60, color="tab:purple")
    ax[0][1].axvline(NO_MOTION_TICKS, color="tab:red", ls="--", lw=1.2,
                     label=f"<{NO_MOTION_TICKS} ticks = no motion")
    ax[0][1].set_yscale("log")
    ax[0][1].set_xlabel("|position change| per command (ticks)")
    ax[0][1].set_ylabel("count (log)")
    ax[0][1].set_title("motion produced per command; the spike at zero is the "
                       "silent-drop population", fontsize=10)
    ax[0][1].legend(fontsize=8)
    ax[0][1].grid(alpha=.3)

    # panel 3: silent drop rate per motor and mode
    labels, values = [], []
    for (mode, m), s in stats.items():
        if s["judged"]:
            labels.append(f"{mode}\nm{m}")
            values.append(100.0 * s["no_motion"] / s["judged"])
    if labels:
        ax[1][0].bar(range(len(labels)), values, color="tab:orange")
        ax[1][0].set_xticks(range(len(labels)))
        ax[1][0].set_xticklabels(labels, fontsize=8)
    ax[1][0].set_ylabel("silent drops (%)")
    ax[1][0].set_title("drop rate by mode and motor — 'all' is the real control "
                       "loop path", fontsize=10)
    ax[1][0].grid(alpha=.3, axis="y")

    # panel 4: achieved cycle rate
    per = col(rows, "period")
    per = per[~np.isnan(per) & (per > 0)]
    if len(per):
        ax[1][1].hist(1.0 / per, bins=60, color="tab:green")
        ax[1][1].axvline(200, color="tab:red", ls="--", lw=1.2,
                         label="200 Hz (configured)")
        ax[1][1].legend(fontsize=8)
    ax[1][1].set_xlabel("achieved cycle rate (Hz)")
    ax[1][1].set_ylabel("count")
    ax[1][1].set_title("write+read cycles the bus actually sustains", fontsize=10)
    ax[1][1].grid(alpha=.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    print(f"plot -> {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
