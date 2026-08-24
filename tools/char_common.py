"""Shared logging and plotting for the motor characterisation series (tests A-D).

Every characterisation test writes a CSV through Recorder and renders a figure
through one of the plot helpers, so the artifacts are comparable across tests and
nothing depends on remembering what a run did.

Written after a week in which several guards were built on unmeasured assumptions
about these servos. The point of this series is to produce the register behaviour
table that should have existed first, so keep the raw CSV — the plots are a view of
it, not a substitute.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


class Recorder:
    """CSV logger, flushed every row.

    Flushing per row is deliberate: the informative runs are the ones that end
    unexpectedly, and a buffered writer loses exactly the tail you need.
    """

    def __init__(self, path: str | Path, fields: Sequence[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = list(fields)
        self._fh = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=self.fields)
        self._w.writeheader()
        self._fh.flush()
        self.t0 = time.time()
        self.rows = 0

    def log(self, **kw) -> None:
        row = {k: "" for k in self.fields}
        row.update({k: v for k, v in kw.items() if k in self.fields})
        if "t" in self.fields and not row.get("t"):
            row["t"] = f"{time.time() - self.t0:.4f}"
        self._w.writerow(row)
        self._fh.flush()
        self.rows += 1

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass


def load(path: str | Path) -> List[dict]:
    """Read a CSV back, skipping torn final lines."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if any(v is None for v in r.values()):
                continue          # partially written row
            rows.append(r)
    return rows


def col(rows: Sequence[dict], key: str) -> np.ndarray:
    out = []
    for r in rows:
        v = r.get(key, "")
        try:
            out.append(np.nan if v in ("", None) else float(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.array(out, dtype=float)


def unwrap_ticks(values: np.ndarray, period: int = 4096) -> np.ndarray:
    """Undo modular wrapping to recover continuous rotation.

    A jump larger than half a period between consecutive samples is assumed to be
    a wrap rather than real motion — valid here because sampling is far faster than
    the servo can travel 2048 ticks.
    """
    v = np.asarray(values, dtype=float)
    out = v.copy()
    offset = 0.0
    for i in range(1, len(v)):
        if np.isnan(v[i]) or np.isnan(v[i - 1]):
            out[i] = v[i] + offset
            continue
        d = v[i] - v[i - 1]
        if d > period / 2:
            offset -= period
        elif d < -period / 2:
            offset += period
        out[i] = v[i] + offset
    return out


def wrap_events(values: np.ndarray, period: int = 4096) -> List[int]:
    """Indices where a wrap was inferred."""
    v = np.asarray(values, dtype=float)
    idx = []
    for i in range(1, len(v)):
        if np.isnan(v[i]) or np.isnan(v[i - 1]):
            continue
        if abs(v[i] - v[i - 1]) > period / 2:
            idx.append(i)
    return idx


THEME = dict(
    raw="tab:blue", unwrapped="tab:purple", cmd="tab:orange",
    marker="tab:red", grid=dict(alpha=0.3),
)


def summarise_position(values: np.ndarray, period: int = 4096) -> Dict[str, object]:
    finite = values[~np.isnan(values)]
    if not len(finite):
        return {"samples": 0}
    return {
        "samples": int(len(finite)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "negative_samples": int((finite < 0).sum()),
        "above_period": int((finite >= period).sum()),
        "wraps": len(wrap_events(values, period)),
    }


def render_position_trace(
    csv_path: str | Path,
    out_path: Optional[str | Path] = None,
    title: str = "",
    period: int = 4096,
    cmd_key: Optional[str] = None,
):
    """Three panels: raw reading, unwrapped rotation, sample-to-sample delta.

    The delta panel is the diagnostic one — wraps appear as isolated points near
    +/-period, real motion clusters near zero.
    """
    import matplotlib
    if out_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load(csv_path)
    if len(rows) < 2:
        raise SystemExit(f"{csv_path}: not enough rows to plot")

    t = col(rows, "t")
    pos = col(rows, "present")
    unw = unwrap_ticks(pos, period)
    d = np.diff(pos)
    wraps = wrap_events(pos, period)
    stats = summarise_position(pos, period)

    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=False)
    fig.suptitle(title or Path(csv_path).name, fontsize=12)

    ax[0].plot(t, pos, lw=1.4, color=THEME["raw"], label="Present_Position (raw)")
    if cmd_key:
        c = col(rows, cmd_key)
        if not np.all(np.isnan(c)):
            ax[0].plot(t, c, lw=1.2, ls="--", color=THEME["cmd"], label=cmd_key)
    for i in wraps:
        ax[0].axvline(t[i], color=THEME["marker"], lw=0.8, alpha=.5)
    ax[0].axhline(0, color="gray", lw=1, alpha=.5)
    ax[0].axhline(period - 1, color="gray", lw=1, alpha=.5, ls=":")
    ax[0].set_ylabel("ticks")
    ax[0].set_title(
        f"raw reading — range {stats.get('min'):.0f}..{stats.get('max'):.0f}, "
        f"{stats.get('negative_samples')} negative, "
        f"{stats.get('above_period')} at/above {period}, "
        f"{stats.get('wraps')} inferred wraps (red lines)", fontsize=10)
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].grid(**THEME["grid"])

    ax[1].plot(t, unw, lw=1.6, color=THEME["unwrapped"])
    for k in range(int(np.nanmin(unw) // period), int(np.nanmax(unw) // period) + 2):
        ax[1].axhline(k * period, color="gray", lw=0.7, alpha=.35)
    ax[1].set_ylabel("ticks (cumulative)")
    ax[1].set_xlabel("time (s)")
    span = np.nanmax(unw) - np.nanmin(unw)
    ax[1].set_title(f"unwrapped rotation — total travel {span:.0f} ticks "
                    f"({span/period:.2f} turns); grey lines are turn boundaries",
                    fontsize=10)
    ax[1].grid(**THEME["grid"])

    ax[2].plot(np.arange(1, len(pos)), d, ".", ms=3, color=THEME["raw"])
    ax[2].axhline(period / 2, color=THEME["marker"], ls="--", lw=1.2,
                  label=f"±{period//2} (wrap threshold)")
    ax[2].axhline(-period / 2, color=THEME["marker"], ls="--", lw=1.2)
    ax[2].set_ylabel("Δ ticks / sample")
    ax[2].set_xlabel("sample index")
    ax[2].set_title("sample-to-sample change — isolated points beyond the dashed "
                    "lines are wraps, not motion", fontsize=10)
    ax[2].legend(fontsize=8)
    ax[2].grid(**THEME["grid"])

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if out_path:
        fig.savefig(out_path, dpi=130)
        print(f"plot -> {out_path}")
    else:
        plt.show()
    return stats


__all__ = [
    "Recorder", "load", "col", "unwrap_ticks", "wrap_events",
    "summarise_position", "render_position_trace",
]
