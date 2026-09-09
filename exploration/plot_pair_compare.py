#!/usr/bin/env python
"""Paired primitives on ONE shared time axis: breathe, and side_side.

    python exploration/plot_pair_compare.py

Both pairs are the same primitive at two settings -- NORMAL_BREATHE_CONFIG is
SlowBreatheConfig(period_s=2.0), SIDE_SIDE_FAST_CONFIG is SideSideConfig with a
shorter sweep -- so the only honest way to show the difference is a common x
axis. On separate axes each fills its panel with the same number of cycles and
the figure shows nothing at all; that misreading is the point of these plots.

The axis is clipped to the SHORTER run. Extending to the longer one leaves half
the top panel empty, and empty space is not evidence: the claim is a rate, so
the strongest frame is the window where both are still running.

Commands come from RecordingController, the harness char_primitive_sweep uses,
so these are the streams execute_behavior actually sends rather than a
re-derivation of the maths.
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
    NORMAL_BREATHE_CONFIG, SLOW_BREATHE_CONFIG, SIDE_SIDE_CONFIG,
    SIDE_SIDE_FAST_CONFIG, MotionBehavior, execute_behavior, side_side_segments)

CAL = {"1": 2662, "2": 2868, "3": 2981}
OUT = HERE / "out"
BOLD, FAINT = "#1a4fa0", ("#9aa5b1", "#c3ccd6")


def capture(name):
    rec = RecordingController(CAL)
    execute_behavior(rec, MotionBehavior.from_action_string(f"<{name}>"),
                     noise_scale=0.0)
    t = np.array([x for x, _ in rec.commands], float)
    t -= t[0]
    return t, {k: np.array([c[k] for _, c in rec.commands], float) for k in CAL}


def panel(ax, t, m, lead, quiet, label):
    """Draw one primitive. `lead` motors are bold, `quiet` ones faint."""
    for k, c in zip(quiet, FAINT):
        ax.plot(t, m[k], color=c, lw=1.2, label=f"motor {k}")
    for k, style in zip(lead, ("-", "--")):
        ax.plot(t, m[k], color=BOLD, lw=2.0, ls=style, label=f"motor {k}{label}")


def figure(pairs, fname, suptitle, lead, quiet):
    data = {n: capture(n) for n, *_ in pairs}
    span = min(t[-1] for t, _ in data.values())

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    for ax, (name, period, state, note) in zip(axes, pairs):
        t, m = data[name]
        panel(ax, t, m, lead, quiet, "")
        ax.axhline(CAL[lead[0]], color="#888", lw=0.7, ls=":")

        # dimension one period, so the difference is measurable not eyeballed
        span_y = max(m[k].max() for k in lead)
        y = CAL[lead[0]] + 0.82 * (span_y - CAL[lead[0]])
        ax.annotate("", xy=(period, y), xytext=(0, y),
                    arrowprops=dict(arrowstyle="<|-|>", color="#c2185b", lw=1.6))
        ax.text(period / 2, y + 30, f"{period:.2f} s".rstrip("0").rstrip("."),
                color="#c2185b", ha="center", fontsize=10, weight="bold")

        ax.set_title(f"{name}   —   {state}   —   {note}   "
                     f"(full run {t[-1]:.1f} s, {len(t)} commands)",
                     loc="left", fontsize=11)
        ax.set_ylabel("motor position (ticks)")
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right", fontsize=8.5, ncol=2, framealpha=0.9)

    # NO inset. One was tried, zooming a single extreme to show the retreat and
    # overshoot, and it covered the very traces it was explaining. The stutter
    # is already legible unaided -- it is the double bump at each peak -- so the
    # zoom was adding occlusion in exchange for nothing.
    axes[-1].set_xlabel("time (s)  —  shared axis, clipped to the shorter run")
    axes[-1].set_xlim(0, span)
    fig.suptitle(suptitle, fontsize=12.5, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / fname
    fig.savefig(p, dpi=190, facecolor="white")
    plt.close(fig)
    print(f"wrote {p}")
    for n, *_ in pairs:
        t, _ = data[n]
        print(f"  {n:16} {t[-1]:5.2f} s  {len(t):4} commands")


def main() -> int:
    figure(
        [("normal_breathe", NORMAL_BREATHE_CONFIG.period_s, "NOTICING",
          f"period {NORMAL_BREATHE_CONFIG.period_s:.0f} s"),
         ("slow_breathe", SLOW_BREATHE_CONFIG.period_s, "ALONE",
          f"period {SLOW_BREATHE_CONFIG.period_s:.0f} s")],
        "breathe_compare.png",
        "Same primitive, two periods: normal_breathe is the identical shape at "
        "double frequency",
        lead=["2"], quiet=["1", "3"])

    # side_side sways along the 60 deg axis, which is PERPENDICULAR to motor 2 --
    # motors 1 and 3 oppose each other and motor 2 travels exactly 0 ticks. That
    # is the whole left/right-versus-nod point, so 1 and 3 lead and 2 is drawn
    # faint precisely to show it sitting still.
    figure(
        [("side_side_fast", SIDE_SIDE_FAST_CONFIG.sweep_s, "EXCITED",
          f"sweep {SIDE_SIDE_FAST_CONFIG.sweep_s:.2f} s, "
          f"amp {SIDE_SIDE_FAST_CONFIG.amplitude}"),
         ("side_side", SIDE_SIDE_CONFIG.sweep_s, "CONTENT",
          f"sweep {SIDE_SIDE_CONFIG.sweep_s:.2f} s, "
          f"amp {SIDE_SIDE_CONFIG.amplitude}")],
        "side_side_compare.png",
        "Same tick-tock, two sweeps: side_side_fast is the identical shape with "
        "more energy",
        lead=["1", "3"], quiet=["2"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
