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
LW = 2.6

# MONOCHROME BY DEFAULT, and the reason is the diagram's own colour scheme:
# hue there encodes DATA TYPE (teal sensor, blue binary, orange discrete,
# magenta continuous). These glyphs describe OPERATIONS -- hysteresis,
# subtraction, quantisation -- which are not a data type, so giving them hue
# would assert a category that does not exist. Emphasis is carried by line
# weight and dash instead, which is free of that encoding.
#
# The circumplex is the one that mattered: four coloured quadrants implied four
# data types, and two of its hues collided with blue and orange used elsewhere.
# Grey tints keep the quadrants distinguishable without claiming anything.
ACCENT = "#1a1a1a"          # set by --colour to the pink perception hue
QUADS = ("#e8e8e8", "#d6d6d6", "#c4c4c4", "#eeeeee")
COLOUR_ACCENT = "#c2185b"
COLOUR_QUADS = ("#f2b544", "#e2574c", "#5b8def", "#57b894")


# Raster resolution for the PNGs. The SVGs are resolution-free and are what
# should actually go in the diagram; the PNGs exist for tools that will not
# take vector input, and those want plenty of headroom because a glyph placed
# at 48 px on screen is 4x that in a retina export and more again in print.
DPI = 400


def canvas(size=2.0):
    fig, ax = plt.subplots(figsize=(size, size), dpi=DPI)
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
    px = int(fig.get_size_inches()[0] * DPI)
    print(f"  {name}.png (~{px}px) / .svg")


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
    ax.plot([-d, d], [0, 0], color=ACCENT, lw=LW + 2.2, solid_capstyle="round")
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    save(fig, "deadband")


def baseline():
    """A wiggly signal, its slow median, and the difference that is used."""
    fig, ax = canvas(size=2.4)
    rng = np.random.default_rng(3)
    t = np.linspace(0, 1, 260)
    drift = 0.30 * np.sin(2.0 * t) + 0.16 * t
    sig = drift + 0.30 * np.sin(15 * t) + 0.05 * rng.normal(size=t.size)
    ax.fill_between(t, drift, sig, color=ACCENT, alpha=0.16, lw=0)
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
    for q, c in enumerate(QUADS, start=1):
        a0 = np.radians([0, 90, 180, 270][q - 1])
        w = np.linspace(a0, a0 + np.pi / 2, 40)
        ax.fill(np.concatenate([[0], np.cos(w)]),
                np.concatenate([[0], np.sin(w)]), color=c, alpha=1.0, lw=0)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=INK, lw=LW))
    ax.plot([-1, 1], [0, 0], color=INK, lw=1.5)
    ax.plot([0, 0], [-1, 1], color=INK, lw=1.5)
    ax.add_patch(plt.Circle((0, 0), 0.34, facecolor="white", alpha=0.95,
                            edgecolor=INK, lw=LW * 0.85,
                            linestyle=(0, (3, 2))))
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    save(fig, "circumplex")


def bandpass():
    """A passband with rejected shoulders: the 0.8-4 Hz rate test.

    Honest about what it abstracts. Nothing is actually FILTERED -- the
    hysteretic half-cycle rate is computed and compared against the band, so
    no signal passes through anything. But "only oscillation in this band
    counts as a gesture" is what the stage means, and the passband glyph says
    that in one shape. Drawn with hard shoulders rather than a smooth
    roll-off, since the test is a comparison and not a filter response: a
    gentle skirt would imply an attenuation that does not exist.
    """
    fig, ax = canvas()
    lo, hi = -0.42, 0.42
    ax.axhline(0, color="#bbb", lw=1.1)
    ax.plot([-1.05, lo], [-0.62, -0.62], color=INK, lw=LW, solid_capstyle="round")
    ax.plot([lo, lo], [-0.62, 0.55], color=INK, lw=LW, solid_capstyle="round")
    ax.plot([lo, hi], [0.55, 0.55], color=INK, lw=LW + 2.2, solid_capstyle="round")
    ax.plot([hi, hi], [0.55, -0.62], color=INK, lw=LW, solid_capstyle="round")
    ax.plot([hi, 1.05], [-0.62, -0.62], color=INK, lw=LW, solid_capstyle="round")
    # the band edges are the whole content, so mark them
    for x in (lo, hi):
        ax.plot([x, x], [-0.86, -0.72], color=INK, lw=1.6,
                solid_capstyle="round")
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.0, 1.0)
    save(fig, "bandpass")


def quantised():
    """A staircase against a smooth ramp: the ROI size rule."""
    fig, ax = canvas()
    x = np.linspace(0, 1, 400)
    ax.plot(x, x, color="#bbb", lw=1.6, ls=(0, (4, 3)))
    ax.step(x, np.round(x * 5) / 5, where="mid", color=INK, lw=LW)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    save(fig, "quantised")


ICONS = [
    ("schmitt", "Schmitt trigger", "enter and leave at different thresholds"),
    ("deadband", "deadband", "output ignores small input"),
    ("baseline", "baseline subtraction", "signal minus its slow median"),
    ("circumplex", "circumplex + neutral", "quadrants, with a dead disc"),
    ("bandpass", "passband", "only 0.8-4 Hz counts as a gesture"),
    ("quantised", "quantised", "snaps to a grid, ignores wobble"),
]


def sheet(dpi: int = 300):
    """Contact sheet: every glyph at full size and at the sizes it must survive.

    Rendered from the saved PNGs rather than redrawing, so what is being judged
    is the actual artefact that goes in the diagram. The small columns are the
    point of the sheet -- a glyph can look immaculate at full size and turn to
    mush at 40 px, which is how the Schmitt arrows were caught rendering as
    dots.
    """
    import matplotlib.image as mpimg

    sizes = [1.0, 0.42, 0.26]           # full, ~64 px, ~40 px in a diagram
    labels = ["full size", "at 64 px", "at 40 px"]
    n = len(ICONS)
    fig = plt.figure(figsize=(2.15 * n, 5.6), dpi=dpi)
    fig.patch.set_facecolor("white")

    for j, (name, title, blurb) in enumerate(ICONS):
        img = mpimg.imread(OUT / f"{name}.png")
        for i, (s, lab) in enumerate(zip(sizes, labels)):
            # place by hand so the small versions are genuinely smaller rather
            # than a full-size image squeezed into a smaller axes box
            w = 0.132 * s
            x = (j + 0.5) / n - w / 2
            y = 0.62 - i * 0.245 - w / 2
            ax = fig.add_axes([x, y, w, w * (2.15 * n) / 5.6])
            ax.imshow(img)
            ax.set_axis_off()
            if j == 0:
                fig.text(0.012, y + w * 0.16, lab, fontsize=8.5,
                         color="#666", va="center")
        fig.text((j + 0.5) / n, 0.90, title, ha="center", fontsize=11.5,
                 color="#111")
        fig.text((j + 0.5) / n, 0.865, blurb, ha="center", fontsize=8.5,
                 color="#777")

    fig.text(0.5, 0.02, "Line art, transparent, PNG + SVG in "
             "exploration/out/icons/ — use the SVG in the diagram",
             ha="center", fontsize=9, color="#888")
    out = OUT.parent / "icon_sheet.png"
    fig.savefig(out, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  sheet -> {out}  ({dpi} dpi)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=DPI, help="icon raster dpi")
    ap.add_argument("--sheet-dpi", type=int, default=300)
    ap.add_argument("--colour", action="store_true",
                    help="use the pink accent and coloured quadrants instead "
                         "of monochrome (see the palette note above)")
    a = ap.parse_args()
    DPI = a.dpi
    if a.colour:
        ACCENT, QUADS = COLOUR_ACCENT, COLOUR_QUADS
    print(f"writing icons -> {OUT}  ({DPI} dpi)")
    schmitt(); deadband(); baseline(); circumplex(); bandpass(); quantised()
    sheet(a.sheet_dpi)
