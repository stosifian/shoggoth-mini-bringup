#!/usr/bin/env python
"""slow_breathe vs normal_breathe on ONE shared time axis.

    python exploration/plot_breathe_compare.py

The two are the same primitive at different periods -- NORMAL_BREATHE_CONFIG is
SlowBreatheConfig(period_s=2.0) and nothing else changes -- so the only honest
way to show the difference is a common x axis. Plotted on their own axes each
looks identical, because each fills its panel with the same number of cycles;
that is exactly the misreading this figure exists to prevent.

Commands are captured through RecordingController, the same harness
char_primitive_sweep uses, so these are the real command streams execute_behavior
would send rather than a re-derivation of the maths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "tools"))

from char_primitive_sweep import RecordingController          # noqa: E402
from shoggoth_mini.control.primitives import (                # noqa: E402
    NORMAL_BREATHE_CONFIG, SLOW_BREATHE_CONFIG, MotionBehavior, execute_behavior)

CAL = {"1": 2662, "2": 2868, "3": 2981}
OUT = HERE / "out"


def capture(name):
    rec = RecordingController(CAL)
    execute_behavior(rec, MotionBehavior.from_action_string(f"<{name}>"),
                     noise_scale=0.0)
    t = np.array([x for x, _ in rec.commands], float)
    t -= t[0]
    return t, {k: np.array([c[k] for _, c in rec.commands], float) for k in CAL}


def main() -> int:
    runs = [
        ("normal_breathe", NORMAL_BREATHE_CONFIG.period_s, "NOTICING"),
        ("slow_breathe", SLOW_BREATHE_CONFIG.period_s, "ALONE"),
    ]
    data = {n: capture(n) for n, _, _ in runs}
    span = max(t[-1] for t, _ in data.values())

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.0), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    for ax, (name, period, state) in zip(axes, runs):
        t, m = data[name]
        # motor 2 carries the breath; 1 and 3 pay out half each, so drawing them
        # faintly shows the coupling without competing with the shape
        for k, c in (("1", "#9aa5b1"), ("3", "#c3ccd6")):
            ax.plot(t, m[k], color=c, lw=1.1)
        ax.plot(t, m["2"], color="#1a4fa0", lw=2.0, label="motor 2 (drives)")
        ax.axhline(CAL["2"], color="#888", lw=0.7, ls=":")

        # mark one period so the reader can measure rather than eyeball
        y = CAL["2"] + 0.80 * (m["2"].max() - CAL["2"])
        ax.annotate("", xy=(period, y), xytext=(0, y),
                    arrowprops=dict(arrowstyle="<|-|>", color="#c2185b", lw=1.6))
        ax.text(period / 2, y + 28, f"{period:.0f} s", color="#c2185b",
                ha="center", fontsize=10, weight="bold")

        cycles = t[-1] / period
        ax.set_title(f"{name}   —   {state}   —   period {period:.0f} s, "
                     f"{cycles:.0f} cycles in {t[-1]:.1f} s   "
                     f"({len(t)} commands)", loc="left", fontsize=11)
        ax.set_ylabel("motor position (ticks)")
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    axes[-1].set_xlabel("time (s)  —  shared axis")
    axes[-1].set_xlim(0, span)
    fig.suptitle("Same primitive, two periods: normal_breathe is the identical "
                 "shape at double frequency", fontsize=12.5, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "breathe_compare.png"
    fig.savefig(p, dpi=190, facecolor="white")
    print(f"wrote {p}")
    for n, per, _ in runs:
        t, m = data[n]
        print(f"  {n:15} {t[-1]:5.1f}s  {len(t):4} cmds  period {per:.1f}s  "
              f"peak +{m['2'].max()-CAL['2']:.0f} ticks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
