#!/usr/bin/env python
"""Animated spectrograms of the yes/no clips, as GIFs for the write-up.

    python exploration/make_sound_gifs.py

GIF rather than mp4 or an <audio> tag, for a boring but decisive reason: GitHub
strips <audio> from repo markdown, and inline video needs a user-attachments URL
rather than a relative path. A GIF is just an image, so it renders anywhere the
rest of the figures do.

The point of animating at all is that the CLAIM is about a contour -- rising
pitch reads affirmative, falling reads negative -- and a static spectrogram
makes the reader trace that themselves. A playhead sweeping in real time turns
it into something you watch happen.

Both clips share one figure and one clock, so the rise and the fall are directly
comparable rather than being two pictures at two scales.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.io import wavfile
from scipy.signal import spectrogram

HERE = Path(__file__).resolve().parent
AUDIO = HERE.parent / "assets" / "audio"
OUT = HERE / "out"
SR = 22050
# Only a playhead moves, so frame rate buys very little here and costs real
# size: a GIF palette dithers the spectrogram gradient, and every frame pays
# for it again. 16 is still a smooth sweep.
FPS = 16
# Measured from the clips themselves: "yes" peaks at ~2.1 kHz, "no" starts at
# ~1.2 kHz and falls. Anything above this is empty spectrum, and every pixel
# spent on it shrinks the contour that is the entire subject of the figure.
FMAX = 2600


def load(path: Path) -> np.ndarray:
    """Decode m4a to mono float via ffmpeg, which reads it and scipy does not."""
    # Decode to a FILE, not a pipe. A piped wav carries an unknown-length
    # header, so scipy warns loudly about hitting EOF early. It is only a
    # warning -- both routes decode byte-identical audio, 22550 samples for
    # yes.m4a -- but a warning that reads like data loss is worse than useless
    # in a figure script, because the next person will believe it.
    #
    # The gap against ffprobe (1.02 s decoded vs 1.07 s reported) is AAC
    # priming and padding in the container, not a short read.
    tmp = OUT / f"._{path.stem}.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path),
                    "-ac", "1", "-ar", str(SR), str(tmp)], check=True)
    _, x = wavfile.read(tmp)
    tmp.unlink()
    x = x.astype(float)
    return x / (np.abs(x).max() or 1.0)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    clips = [("yes", "#2e7d32", "rising  →  affirmative"),
             ("no", "#c62828", "falling  →  negative")]
    data = {}
    for name, _, _ in clips:
        x = load(AUDIO / f"{name}.m4a")
        f, t, S = spectrogram(x, fs=SR, nperseg=512, noverlap=384,
                              scaling="spectrum")
        keep = f <= FMAX
        data[name] = (t, f[keep], 10 * np.log10(S[keep] + 1e-12), len(x) / SR)

    span = max(d[3] for d in data.values())
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    fig.patch.set_facecolor("white")
    heads = []

    for ax, (name, colour, blurb) in zip(axes, clips):
        t, f, S, dur = data[name]
        vmax = S.max()
        ax.pcolormesh(t, f / 1000.0, S, shading="gouraud",
                      cmap="magma", vmin=vmax - 55, vmax=vmax)
        ax.set_ylabel("kHz")
        ax.set_title(f'"{name}"   —   {blurb}   —   {dur:.2f} s',
                     loc="left", fontsize=11, color=colour)
        # the clip ends before the shared window does; mark where
        if dur < span - 1e-3:
            ax.axvspan(dur, span, color="white", alpha=0.55, lw=0)
            ax.axvline(dur, color="#555", lw=0.9, ls=":")
        heads.append(ax.axvline(0, color="white", lw=1.8, alpha=0.9))

    axes[-1].set_xlabel("time (s)  —  shared axis")
    axes[-1].set_xlim(0, span)
    fig.suptitle("FM/AM synth clips: the pitch contour is the message",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    frames = int(round(span * FPS)) + 1

    def step(i):
        for h in heads:
            h.set_xdata([i / FPS, i / FPS])
        return heads

    anim = FuncAnimation(fig, step, frames=frames, interval=1000 / FPS,
                         blit=True)
    p = OUT / "sound_spectrograms.gif"
    anim.save(p, writer=PillowWriter(fps=FPS), dpi=92)
    plt.close(fig)
    print(f"wrote {p}  ({frames} frames, {span:.2f} s, {p.stat().st_size/1e6:.2f} MB)")
    for name, _, _ in clips:
        print(f"  {name:4} {data[name][3]:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
