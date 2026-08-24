"""Live rolling-window view of closed-loop telemetry, while the loop is running.

Tails the CSV that `closed_loop --log-csv` writes, so it needs no access to the
camera or the motor bus — run it in a second terminal alongside the controller
and there is no contention. (Running debug-perception alongside is impossible;
the camera is exclusive. This is the way to watch live.)

Three panels, sized for glancing at rather than reading:
  1. detections — per-eye and combined, last N seconds
  2. tracking error — |target − tip| and its axis breakdown, the number the
     controller exists to drive to zero
  3. policy action — with the saturation band marked, since a cursor pinned at
     its limit means the target is unreachable, not that the policy is noisy

  python tools/plot_loop_live.py /tmp/loop.csv
  python tools/plot_loop_live.py /tmp/loop.csv --window 20 --refresh 4
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

LEGEND_KW = dict(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1,
                 borderaxespad=0.0, frameon=False, fontsize=8)


def read_rows(path):
    """Read the CSV defensively — the writer may be mid-line when we look."""
    try:
        with open(path, "r", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except (FileNotFoundError, OSError):
        return []
    good = []
    for r in rows:
        # A torn final line still parses as a row, but DictReader fills the
        # missing trailing fields with None — which float() raises TypeError on,
        # not ValueError. Reject any row that is not fully written.
        if r.get("t") in (None, "") or any(v is None for v in r.values()):
            continue
        try:
            float(r["t"])
        except (TypeError, ValueError):
            continue
        good.append(r)
    return good


def col(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        try:
            out.append(np.nan if v in ("", None) else float(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.array(out, dtype=float)


def vec(rows, pfx):
    return np.stack([col(rows, f"{pfx}_{a}") for a in "xyz"], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--window", type=float, default=10.0, help="seconds shown")
    ap.add_argument("--refresh", type=float, default=5.0, help="redraws per second")
    ap.add_argument("--reach-mm", type=float, default=50.0)
    args = ap.parse_args()

    fig, ax = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    fig.canvas.manager.set_window_title(f"live — {args.csv.name}")
    # Created ONCE. ax.clear() does not remove twin axes, so building this inside
    # draw() would leak a new axes object every frame and slowly starve the redraw.
    a2 = ax[2].twinx()

    def draw(_frame):
        rows = read_rows(args.csv)
        for a in list(ax) + [a2]:
            a.clear()

        if len(rows) < 2:
            ax[0].set_title(f"waiting for data in {args.csv} …")
            return

        t_all = col(rows, "t")
        t_end = np.nanmax(t_all)
        t0 = t_end - args.window
        keep = t_all >= t0
        rows = [r for r, k in zip(rows, keep) if k]
        if len(rows) < 2:
            return

        t = col(rows, "t")
        tip, tgt = vec(rows, "tip"), vec(rows, "target")
        act = np.stack([col(rows, "action_x"), col(rows, "action_y")], axis=1)
        ticks = np.stack([col(rows, f"ticks_{i}") for i in (1, 2, 3)], axis=1)

        rate = len(rows) / max(t[-1] - t[0], 1e-6)
        acting = np.nansum(col(rows, "detected"))
        fig.suptitle(f"{args.csv.name} — last {args.window:.0f} s · "
                     f"{rate:.1f} Hz · acting {100*acting/len(rows):.0f}% · "
                     f"t = {t_end:.1f} s", fontsize=11)

        # --- 1. detections ---
        tip_ok, tgt_ok = ~np.isnan(tip[:, 0]), ~np.isnan(tgt[:, 0])
        lanes = [(3.4, "tip_l", "tab:blue", "tip L"), (3.9, "tip_r", "tab:cyan", "tip R"),
                 (-0.7, "fing_l", "tab:orange", "finger L"),
                 (-1.2, "fing_r", "tab:red", "finger R")]
        have_eyes = not np.all(np.isnan(col(rows, "fing_l")))
        ax[0].fill_between(t, 2, 3, where=tip_ok, step="mid", color="tab:blue", alpha=.75)
        ax[0].fill_between(t, 0, 1, where=tgt_ok, step="mid", color="tab:orange", alpha=.75)
        if have_eyes:
            for y, key, c, _ in lanes:
                ax[0].fill_between(t, y, y + 0.4, where=col(rows, key) == 1,
                                   step="mid", color=c, alpha=.8)
            ax[0].set_yticks([-1.0, -0.5, 0.5, 2.5, 3.6, 4.1])
            ax[0].set_yticklabels(["finger R", "finger L", "target 3D", "tip 3D",
                                   "tip L", "tip R"], fontsize=8)
            ax[0].set_ylim(-1.4, 4.5)
        else:
            ax[0].set_yticks([0.5, 2.5])
            ax[0].set_yticklabels(["target", "tip"], fontsize=8)
            ax[0].set_ylim(-0.3, 3.3)
        ax[0].set_title(f"detections — tip {100*tip_ok.mean():.0f}%, "
                        f"target {100*tgt_ok.mean():.0f}%", fontsize=10)
        ax[0].grid(alpha=.3, axis="x")

        # --- 2. tracking error ---
        err = tgt - tip
        dist = np.linalg.norm(err, axis=1) * 1000.0
        for i, nm in enumerate("XYZ"):
            ax[1].plot(t, err[:, i] * 1000.0, lw=1.2, color=f"C{i}", alpha=.65,
                       label=f"{nm} error")
        ax[1].plot(t, dist, lw=2.2, color="tab:purple", label="|error|")
        ax[1].axhline(0, color="gray", lw=1, alpha=.5)
        ax[1].axhline(args.reach_mm, color="red", ls="--", lw=1.5,
                      label=f"{args.reach_mm:.0f} mm")
        fin = dist[~np.isnan(dist)]
        med = f"{np.median(fin):.0f}" if len(fin) else "—"
        near = f"{100*np.mean(fin < args.reach_mm):.0f}" if len(fin) else "—"
        ax[1].set_title(f"tracking error — median {med} mm, "
                        f"within {args.reach_mm:.0f} mm {near}% of frames", fontsize=10)
        ax[1].set_ylabel("mm")
        ax[1].legend(**LEGEND_KW)
        ax[1].grid(alpha=.3)

        # --- 3. action + ticks ---
        ax[2].plot(t, act[:, 0], lw=1.6, label="action x")
        ax[2].plot(t, act[:, 1], lw=1.6, label="action y")
        finite_act = act[~np.isnan(act).any(axis=1)]
        if len(finite_act) > 3:
            lim = np.nanmax(np.abs(finite_act))
            # A cursor pinned near its own extreme is saturation, not noise.
            ax[2].axhspan(0.95 * lim, lim * 1.05, color="tab:red", alpha=.07)
            ax[2].axhspan(-lim * 1.05, -0.95 * lim, color="tab:red", alpha=.07)
        ax[2].axhline(0, color="gray", lw=1, alpha=.5)
        ax[2].set_ylabel("cursor")
        ax[2].set_xlabel("time (s)")
        ax[2].grid(alpha=.3)
        for i in range(3):
            a2.plot(t, ticks[:, i], lw=1.0, ls=":", alpha=.5, label=f"motor {i+1}")
        a2.set_ylabel("ticks")
        h1, l1 = ax[2].get_legend_handles_labels()
        h2, l2 = a2.get_legend_handles_labels()
        ax[2].legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(1.06, 1.0),
                     ncol=1, frameon=False, fontsize=8)
        ax[2].set_title("policy action (shaded = saturated) and motor ticks",
                        fontsize=10)

        ax[0].set_xlim(t0, max(t_end, t0 + args.window))
        fig.tight_layout(rect=[0, 0, 0.86, 0.96])

    # cache_frame_data=False: this is an unbounded live stream, not a fixed animation
    _anim = FuncAnimation(fig, draw, interval=int(1000 / args.refresh),
                          cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    raise SystemExit(main())
