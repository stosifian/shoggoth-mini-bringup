#!/usr/bin/env python
"""Micro-icons for the Shoggoth Mirror system diagram.

Each replaces a text annotation with the standard graphical form for the thing
it describes -- the transfer-function idiom used in control and signal
diagrams, which a reader parses without a caption. Line art only, transparent
background, no text inside the glyph, so they drop into a box at any size.

    python exploration/make_diagram_icons.py

Writes both PNG (drop-in) and SVG (scales in the diagram tool) to
exploration/out/icons/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "out" / "icons"
INK = "#1a1a1a"
ACCENT = "#c2185b"          # matches the pink perception boxes
LW = 2.6


def canvas(size=2.0):
    fig, ax = plt.subplots(figsize=(size, size), dpi=160)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.patch.set_alpha(0.0)
    return fig, ax


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", transparent=True,
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  {name}.png / .svg")


def schmitt():
    """The standard hysteresis glyph: two transitions, arrows for direction."""
    fig, ax = canvas()
    lo, hi = -0.45, 0.45
    ax.plot([-1, hi], [-0.6, -0.6], color=INK, lw=LW, solid_capstyle="round")
    ax.plot([hi, hi], [-0.6, 0.6], color=INK, lw=LW, solid_capstyle="round")
    ax.plot([hi, 1], [0.6, 0.6], color=INK, lw=LW, solid_capstyle="round")
    ax.plot([lo, 1], [0.6, 0.6], color=INK, lw=LW, solid_capstyle="round")
    ax.plot([lo, lo], [0.6, -0.6], color=INK, lw=LW, solid_capstyle="round")
    ax.plot([-1, lo], [-0.6, -0.6], color=INK, lw=LW, solid_capstyle="round")
    # Direction arrows -- the whole point of the symbol, and the part that says
    # "entering and leaving happen at different thresholds". They need real
    # length: drawn short they render as dots at icon size and the glyph
    # degrades into a plain step.
    ax.annotate("", xy=(0.30, -0.6), xytext=(-0.50, -0.6),
                arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.5",
                                color=ACCENT, lw=LW, shrinkA=0, shrinkB=0))
    ax.annotate("", xy=(-0.30, 0.6), xytext=(0.50, 0.6),
                arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.5",
                                color=ACCENT, lw=LW, shrinkA=0, shrinkB=0))
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.0, 1.0)
    save(fig, "schmitt")


def deadband():
    """Transfer curve with a flat zone: output ignores small input."""
    fig, ax = canvas()
    x = np.linspace(-1, 1, 400)
    d = 0.35
    y = np.where(np.abs(x) < d, 0.0, np.sign(x) * (np.abs(x) - d) / (1 - d))
    ax.axhline(0, color="#bbb", lw=1.1)
    ax.axvline(0, color="#bbb", lw=1.1)
    ax.plot(x, y, color=INK, lw=LW, solid_capstyle="round")
    ax.plot([-d, d], [0, 0], color=ACCENT, lw=LW + 1.4, solid_capstyle="round")
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    save(fig, "deadband")


def baseline():
    """A wiggly signal, its slow median, and the difference that is used."""
    fig, ax = canvas(size=2.4)
    rng = np.random.default_rng(3)
    t = np.linspace(0, 1, 260)
    drift = 0.30 * np.sin(2.0 * t) + 0.16 * t
    sig = drift + 0.30 * np.sin(15 * t) + 0.05 * rng.normal(size=t.size)
    ax.fill_between(t, drift, sig, color=ACCENT, alpha=0.22, lw=0)
    ax.plot(t, sig, color=INK, lw=LW * 0.82, solid_capstyle="round")
    ax.plot(t, drift, color=ACCENT, lw=LW, ls=(0, (4, 2.6)))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.55, 0.95)
    save(fig, "baseline")


def circumplex():
    """Two axes, four quadrants, a deadband disc at the origin.

    Carries three facts at once: the axes are independent, the four states are
    quadrants of them, and the middle is neutral rather than a fifth state.
    """
    fig, ax = canvas(size=2.2)
    for q, c in ((1, "#f2b544"), (2, "#e2574c"), (3, "#5b8def"), (4, "#57b894")):
        a0 = np.radians([0, 90, 180, 270][q - 1])
        w = np.linspace(a0, a0 + np.pi / 2, 40)
        ax.fill(np.concatenate([[0], np.cos(w)]),
                np.concatenate([[0], np.sin(w)]), color=c, alpha=0.30, lw=0)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=INK, lw=LW))
    ax.plot([-1, 1], [0, 0], color=INK, lw=1.5)
    ax.plot([0, 0], [-1, 1], color=INK, lw=1.5)
    ax.add_patch(plt.Circle((0, 0), 0.34, facecolor="white", alpha=0.95,
                            edgecolor=INK, lw=LW * 0.85,
                            linestyle=(0, (3, 2))))
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    save(fig, "circumplex")


def quantised():
    """A staircase against a smooth ramp: the ROI size rule."""
    fig, ax = canvas()
    x = np.linspace(0, 1, 400)
    ax.plot(x, x, color="#bbb", lw=1.6, ls=(0, (4, 3)))
    ax.step(x, np.round(x * 5) / 5, where="mid", color=INK, lw=LW)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    save(fig, "quantised")


if __name__ == "__main__":
    print("writing icons ->", OUT)
    schmitt(); deadband(); baseline(); circumplex(); quantised()
