"""Plot closed-loop telemetry written by `closed_loop --log-csv`.

You cannot watch debug-perception while the controller runs (the camera is
exclusive to one process), so the CSV is the only record of what the loop saw.
This turns it into the four views that actually diagnose jerk:

  1. Detection timeline  — when each of tip/target was present or missing
  2. 3D positions        — tip and target over time, per axis
  3. Tracking error      — distance from tip to target, and its per-axis parts.
                           This is the thing the controller exists to drive to zero,
                           so a flat trace well above zero means it is not converging
                           (often because the target sits outside the reachable
                           volume), and the axis breakdown says which direction is
                           to blame.
  4. Action + motor ticks— what the policy asked for and what the motors were told

Run:
  python tools/plot_loop_csv.py /tmp/loop2.csv
  python tools/plot_loop_csv.py /tmp/loop2.csv --reach-mm 40 --save plot.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Legends live outside the axes on the right, stacked vertically, so they never
# sit on top of the data.
LEGEND_KW = dict(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1,
                 borderaxespad=0.0, frameon=False)


def load(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"{path} is empty")

    def col(key, dtype=float):
        out = []
        for r in rows:
            v = r.get(key, "")
            out.append(np.nan if v == "" else dtype(v))
        return np.array(out, dtype=float)

    d = {
        "t": col("t"),
        "dt": col("dt"),
        "detected": col("detected"),
        "lost": col("lost_counter"),
    }
    for pfx in ("tip", "target"):
        d[pfx] = np.stack([col(f"{pfx}_{a}") for a in "xyz"], axis=1)
    d["action"] = np.stack([col("action_x"), col("action_y")], axis=1)
    # Per-eye flags are optional: older CSVs predate them.
    for k in ("tip_l", "tip_r", "fing_l", "fing_r"):
        d[k] = col(k)
    d["ticks"] = np.stack([col(f"ticks_{i}") for i in (1, 2, 3)], axis=1)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--reach-mm", type=float, default=50.0,
                    help="distance below which the tip counts as having reached the "
                         "target; drawn as a reference line on the error plot")
    ap.add_argument("--save", type=Path, default=None, help="write PNG instead of showing")
    args = ap.parse_args()

    d = load(args.csv)
    t = d["t"]
    n = len(t)
    acting = int(np.nansum(d["detected"]))
    rate = 1.0 / np.nanmean(d["dt"][1:]) if n > 1 else float("nan")

    fig, ax = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f"{args.csv.name} — {n} rows, {acting} acting "
                 f"({100*acting/n:.0f}%), {rate:.1f} Hz", fontsize=12)

    # 1. detection timeline
    tip_ok = ~np.isnan(d["tip"][:, 0])
    tgt_ok = ~np.isnan(d["target"][:, 0])
    ax[0].fill_between(t, 2, 3, where=tip_ok, step="mid", color="tab:blue", alpha=.75)
    ax[0].fill_between(t, 0, 1, where=tgt_ok, step="mid", color="tab:orange", alpha=.75)
    # Per-eye rows, when the CSV has them: a triangulated point needs BOTH eyes,
    # so "seen in one eye only" is a distinct failure from "not seen at all".
    have_eyes = not np.all(np.isnan(d["fing_l"]))
    if have_eyes:
        for row_y, key, colr in ((3.4, "tip_l", "tab:blue"), (3.9, "tip_r", "tab:cyan"),
                                 (-0.7, "fing_l", "tab:orange"),
                                 (-1.2, "fing_r", "tab:red")):
            ax[0].fill_between(t, row_y, row_y + 0.4, where=d[key] == 1,
                               step="mid", color=colr, alpha=.8)
        ax[0].set_yticks([-1.0, -0.5, 0.5, 2.5, 3.6, 4.1])
        ax[0].set_yticklabels(["finger R eye", "finger L eye", "target\n(3D)",
                               "tip\n(3D)", "tip L eye", "tip R eye"])
        ax[0].set_ylim(-1.4, 4.5)
    else:
        ax[0].set_yticks([0.5, 2.5])
        ax[0].set_yticklabels(["target\n(finger)", "tip"])
        ax[0].set_ylim(-0.3, 3.3)
    ax[0].set_title(f"detections present  —  tip {100*tip_ok.mean():.0f}%, "
                    f"target {100*tgt_ok.mean():.0f}%  (gaps = dropouts)")
    ax[0].grid(alpha=.3, axis="x")

    # 2. positions
    for arr, name, ls in ((d["tip"], "tip", "-"), (d["target"], "target", "--")):
        for i, axis_name in enumerate("XYZ"):
            ax[1].plot(t, arr[:, i], ls, lw=1.5,
                       color=f"C{i}", label=f"{name} {axis_name}", alpha=.85)
    ax[1].set_ylabel("metres")
    ax[1].set_title("3D positions (solid = tip, dashed = target)")
    ax[1].legend(**LEGEND_KW, fontsize=8)
    ax[1].grid(alpha=.3)

    # 3. tracking error — what the controller is actually trying to null out
    err = d["target"] - d["tip"]              # NaN wherever either point is missing
    dist_mm = np.linalg.norm(err, axis=1) * 1000.0
    finite = dist_mm[~np.isnan(dist_mm)]

    for i, axis_name in enumerate("XYZ"):
        ax[2].plot(t, err[:, i] * 1000.0, lw=1.2, color=f"C{i}", alpha=.65,
                   label=f"{axis_name} error")
    ax[2].plot(t, dist_mm, lw=2.0, color="tab:purple", label="|error|")
    ax[2].axhline(0, color="currentColor" if False else "gray", lw=1, alpha=.5)
    ax[2].axhline(args.reach_mm, color="red", ls="--", lw=1.6,
                  label=f"{args.reach_mm:.0f} mm = reached")

    if len(finite):
        ax[2].set_title(
            f"tracking error (target − tip) — median |error| {np.median(finite):.0f} mm, "
            f"closest {finite.min():.0f} mm, within {args.reach_mm:.0f} mm "
            f"{100*np.mean(finite < args.reach_mm):.0f}% of frames"
        )
    else:
        ax[2].set_title("tracking error (target − tip)")
    ax[2].set_ylabel("mm")
    ax[2].legend(**LEGEND_KW, fontsize=8)
    ax[2].grid(alpha=.3)

    # 4. action and motor ticks
    ax[3].plot(t, d["action"][:, 0], lw=1.5, label="action x")
    ax[3].plot(t, d["action"][:, 1], lw=1.5, label="action y")
    ax[3].set_ylabel("cursor")
    ax[3].set_xlabel("time (s)")
    ax[3].grid(alpha=.3)
    ax3b = ax[3].twinx()
    for i in range(3):
        ax3b.plot(t, d["ticks"][:, i], lw=1.2, alpha=.55, ls=":",
                  label=f"motor {i+1}")
    ax3b.set_ylabel("motor ticks (dotted)")
    h1, l1 = ax[3].get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    # Panel 4 has a twin axis on the right, so its legend needs extra clearance
    # or it lands on top of the "motor ticks" label and tick numbers.
    ax[3].legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(1.06, 1.0),
                 ncol=1, borderaxespad=0.0, frameon=False, fontsize=8)
    ax[3].set_title("policy action (solid, left) and commanded motor ticks "
                    "(dotted, right)")

    fig.tight_layout(rect=[0, 0, 0.86, 0.97])
    if args.save:
        fig.savefig(args.save, dpi=130)
        print(f"wrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    raise SystemExit(main())
