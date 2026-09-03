"""Parametric motion playground -- headless generator + per-mood plots.

DESIGN/RESEARCH (see NEXT_PHASE_EXPLORATION.md). Imports the POC's real
2D->tendon mapping read-only; does NOT modify POC code.

Pipeline:
    affect (arousal, valence, tension)          # Step 1: interpretable dials
      --g-->  motion params                      # Step 3: the creative mapping
      --generate-->  (c_x, c_y, a)(t)            # Step 4: phrase-based trajectory
      --embody-->  tendon lengths -> (sim)       # Step 5: convert + axial channel

v2 -- distinctiveness fix.  Moods 3-6 collapsed because arousal drove all the
*visible* variance while valence/tension were wired to near-invisible channels,
and every mood used ONE oscillator archetype.  Expression lives in the temporal
*phrasing*, not the magnitude, so the generator now exposes qualitative
mechanisms the affect axes gate:
  * phrasing (move <-> hold / freeze-then-dart)  <- tension, arousal
  * reorientation / scanning of bend direction    <- arousal, valence
  * tremor (braced trembling)                      <- tension
  * roundness/openness vs small/angular/withdrawn  <- valence
plus amplitude headroom so moods don't die at the saturation rail.

Run:  .venv/bin/python exploration/motion_playground.py
"""

import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shoggoth_mini.control.geometry import convert_2d_cursor_to_target_lengths

# --- ground truth (verified against configs / generate_mujoco_xml) ------------
ACT_LOW, ACT_HIGH, BASELINE, MAX_2D_MAG = 0.12, 0.34, 0.23, 1.0
AXIAL_SAFE_M = 0.08          # per-tendon common-mode bound (slack/self-collision safety)
REACH_MAX = 0.85             # cursor-magnitude headroom (leave room before saturation)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

_baseline = np.full(3, BASELINE, np.float32)
_low = np.full(3, ACT_LOW, np.float32)
_high = np.full(3, ACT_HIGH, np.float32)


# =============================================================================
# Step 1 -- affect space.  arousal in [0,1], valence in [-1,1], tension in [0,1]
# =============================================================================
@dataclass
class Affect:
    arousal: float   # 0 calm .. 1 energised
    valence: float   # -1 negative .. +1 positive
    tension: float   # 0 relaxed .. 1 braced


MOODS = {
    "sleepy":  Affect(arousal=0.10, valence=0.20, tension=0.10),
    "content": Affect(arousal=0.30, valence=0.80, tension=0.10),
    "curious": Affect(arousal=0.55, valence=0.50, tension=0.40),
    "alert":   Affect(arousal=0.70, valence=0.00, tension=0.75),
    "wary":    Affect(arousal=0.60, valence=-0.50, tension=0.85),
    "excited": Affect(arousal=0.95, valence=0.90, tension=0.50),
}


# =============================================================================
# Step 2/3 -- motion parameters + the hand-authored mapping g: affect -> params
# =============================================================================
@dataclass
class MotionParams:
    move_rate: float      # gestures per second
    hold_fraction: float  # fraction of each phrase spent frozen/holding [0..1]
    reach: float          # target cursor magnitude [0..REACH_MAX]
    scan: float           # std (rad) of direction reorientation between gestures
    dir_commit: float     # how strongly targets pull to the bias direction [0..1]
    bias: np.ndarray      # static directional centre (perk up / droop / withdraw)
    attack: float         # easing snappiness (0 smooth .. 1 fast/sudden)
    roundness: float      # 0 direct/angular .. 1 rounded/overshoot (bouncy)
    tremor_amp: float     # braced-trembling amplitude (cursor units)
    tremor_freq: float
    breath_amp: float     # axial breathing amplitude (m)
    breath_freq: float
    axial_static: float   # static axial offset (m): - contract / + extend
    recoil: float         # axial contraction coupled to gesture speed [0..1]


def affect_to_params(a: Affect) -> MotionParams:
    """g: the creative core. Hand-tuned; each affect axis gates a *qualitative*
    mechanism so the moods separate, not just scale. Tune by watching."""
    ar, val, ten = a.arousal, a.valence, a.tension
    neg = max(0.0, -val)              # how negative the valence is
    openness = (val + 1.0) / 2.0      # 0 (neg) .. 1 (pos)
    return MotionParams(
        # phrasing: arousal -> more gestures; tension/low-arousal -> more freezing
        move_rate=0.30 + 1.10 * ar,
        hold_fraction=float(np.clip(0.12 + 0.60 * ten - 0.28 * ar, 0.03, 0.80)),
        # size: arousal + openness, minus withdrawal; capped for headroom
        reach=float(np.clip((0.18 + 0.50 * ar) * (0.65 + 0.50 * openness)
                            - 0.20 * neg, 0.08, REACH_MAX)),
        # reorientation: aroused -> darts widely; withdrawn -> narrow/fixed.
        # (kept deliberate, not frantic -- energy comes from reach+attack, not
        #  from flipping direction many times a second)
        scan=float(np.clip(0.25 + 0.85 * ar - 1.00 * neg, 0.12, 1.6)),
        # commitment to bias: calm commits to a posture; wary guards one way
        dir_commit=float(np.clip(0.35 + 0.40 * (1 - ar) + 0.30 * neg, 0.2, 0.9)),
        # posture: valence perks up / droops; low energy droops; neg withdraws sideways
        bias=np.array([-0.15 * neg, 0.30 * val - 0.20 * (1 - ar)], np.float32),
        # sudden vs smooth (Laban Time); tense+aroused -> snappy
        attack=float(np.clip(0.15 + 0.45 * ar + 0.35 * ten, 0, 1)),
        # rounded/bouncy when positive & relaxed; angular when tense
        roundness=float(np.clip(openness - 0.40 * ten, 0, 1)),
        # trembling only when genuinely braced (quadratic gate); coherent low-freq
        # quiver (see generate_trajectory), not white noise
        tremor_amp=float(np.clip(0.05 * ten * ten, 0, 0.08)),
        tremor_freq=4.0,
        breath_amp=0.012 + 0.030 * ar,
        breath_freq=0.25 + 0.25 * ar,
        axial_static=float(np.clip(-0.07 * ten + 0.05 * neg,
                                   -AXIAL_SAFE_M, AXIAL_SAFE_M)),
        recoil=float(np.clip(0.15 + 0.45 * ar + 0.35 * ten, 0, 1)),
    )


# =============================================================================
# Step 4 -- phrase-based trajectory generator: params -> (c_x, c_y, a)(t)
# =============================================================================
def _ease(u: np.ndarray, attack: float, roundness: float) -> np.ndarray:
    """Ease-out with adjustable speed (attack) blended toward an overshoot-and-
    settle (roundness -> bouncy). u in [0,1]."""
    k = 1.0 + 3.0 * attack
    base = 1.0 - (1.0 - u) ** k                     # fast attack, settle
    c1 = 1.70158 * roundness
    back = 1.0 + (c1 + 1.0) * (u - 1.0) ** 3 + c1 * (u - 1.0) ** 2   # overshoot
    return (1.0 - roundness) * base + roundness * back


def _lowpass(x: np.ndarray, fps: int, cutoff_hz: float) -> np.ndarray:
    """One-pole low-pass: removes buzzy high-freq jitter, keeps gesture-scale
    motion. Also smooths the setpoint the tentacle chases (less physical ring)."""
    if cutoff_hz <= 0:
        return x
    dt = 1.0 / fps
    alpha = dt / (1.0 / (2.0 * np.pi * cutoff_hz) + dt)   # in (0,1]
    try:
        from scipy.signal import lfilter
        return lfilter([alpha], [1.0, -(1.0 - alpha)], x)
    except Exception:                                     # manual fallback
        y = np.empty_like(x); acc = x[0]
        for i, v in enumerate(x):
            acc += alpha * (v - acc); y[i] = acc
        return y


def generate_trajectory(p: MotionParams, duration: float = 8.0, fps: int = 50,
                        seed: int = 0) -> dict:
    """Sequence of eased 'gesture then hold' phrases with reorientation, tremor,
    breathing and speed-coupled axial recoil."""
    rng = np.random.default_rng(seed)
    n = int(duration * fps)
    t = np.arange(n) / fps
    cx = np.zeros(n); cy = np.zeros(n)

    bias_angle = (np.arctan2(p.bias[1], p.bias[0])
                  if np.hypot(*p.bias) > 1e-6 else np.pi / 2)
    prev_angle = bias_angle
    prev_target = np.zeros(2)
    k = 0
    while k < n:
        period = 1.0 / max(p.move_rate, 1e-3)
        n_move = max(1, int(period * (1 - p.hold_fraction) * fps))
        n_hold = int(period * p.hold_fraction * fps)

        angle = ((1 - p.dir_commit) * prev_angle + p.dir_commit * bias_angle
                 + rng.normal(0, p.scan))
        mag = np.clip(p.reach * (1 + rng.uniform(-0.12, 0.12)), 0, REACH_MAX)
        target = mag * np.array([np.cos(angle), np.sin(angle)]) + p.bias
        tm = np.hypot(*target)
        if tm > REACH_MAX:
            target *= REACH_MAX / tm
        prev_angle = angle

        if n_move > 0:                                   # eased move
            u = (np.arange(1, n_move + 1)) / n_move
            e = _ease(u, p.attack, p.roundness)[:, None]
            seg = prev_target[None, :] + (target - prev_target)[None, :] * e
            m = min(n_move, n - k)
            cx[k:k + m] = seg[:m, 0]; cy[k:k + m] = seg[:m, 1]; k += m
        if k < n and n_hold > 0:                         # freeze / hold
            m = min(n_hold, n - k)
            cx[k:k + m] = target[0]; cy[k:k + m] = target[1]; k += m
        prev_target = target

    # braced trembling: a small COHERENT quiver (Lissajous), not white noise
    if p.tremor_amp > 0:
        ph = rng.uniform(0, 2 * np.pi)
        cx += p.tremor_amp * np.sin(2 * np.pi * p.tremor_freq * t + ph)
        cy += p.tremor_amp * np.sin(2 * np.pi * p.tremor_freq * 1.13 * t + ph + 1.7)

    # low-pass to kill residual buzz while preserving the gesture phrasing.
    # snappier moods keep a little more high-freq (higher cutoff), capped low.
    cutoff = 4.0 + 2.5 * p.attack
    cx = _lowpass(cx, fps, cutoff)
    cy = _lowpass(cy, fps, cutoff)

    mag = np.hypot(cx, cy); over = mag > MAX_2D_MAG
    cx[over] *= MAX_2D_MAG / mag[over]; cy[over] *= MAX_2D_MAG / mag[over]

    # axial: static offset + breathing - recoil (contract) during fast gestures
    speed = np.hypot(np.gradient(cx), np.gradient(cy)) * fps
    speed_n = speed / (speed.max() + 1e-6)
    a = (p.axial_static + p.breath_amp * np.sin(2 * np.pi * p.breath_freq * t)
         - p.recoil * 0.06 * speed_n)
    a = _lowpass(a, fps, cutoff)
    a = np.clip(a, -AXIAL_SAFE_M, AXIAL_SAFE_M)

    return {"t": t, "cx": cx, "cy": cy, "a": a}


# =============================================================================
# Step 5 -- embodiment: (c_x, c_y, a) -> 3 tendon lengths (for sim / plots)
# =============================================================================
def cursor_to_tendons(cx: float, cy: float, a: float) -> np.ndarray:
    """2D bend (real POC mapping) + axial common-mode channel, clipped to box."""
    lengths = convert_2d_cursor_to_target_lengths(
        np.array([cx, cy], np.float32), _baseline, _low, _high, MAX_2D_MAG)
    return np.clip(lengths + a, ACT_LOW, ACT_HIGH)


def tendon_series(tr: dict) -> np.ndarray:
    """(N,3) tendon lengths for a whole trajectory."""
    return np.array([cursor_to_tendons(cx, cy, a)
                     for cx, cy, a in zip(tr["cx"], tr["cy"], tr["a"])])


def main() -> None:
    print("Parametric motion playground v2 -- affect -> motion params\n")
    hdr = ("mood", "move_r", "hold_f", "reach", "scan", "commit", "attack",
           "round", "tremor")
    print("  " + " ".join(f"{h:>7}" for h in hdr))
    print("  " + "-" * 72)
    trajs = {}
    for name, aff in MOODS.items():
        p = affect_to_params(aff)
        print(f"  {name:>7} {p.move_rate:7.2f} {p.hold_fraction:7.2f} "
              f"{p.reach:7.2f} {p.scan:7.2f} {p.dir_commit:7.2f} "
              f"{p.attack:7.2f} {p.roundness:7.2f} {p.tremor_amp:7.3f}")
        trajs[name] = generate_trajectory(p, seed=42)

    print("\n  safety check (tendon length range commanded, m):")
    for name, tr in trajs.items():
        L = tendon_series(tr)
        rail = "  <-- at floor" if L.min() <= ACT_LOW + 1e-4 else ""
        print(f"    {name:>7}: [{L.min():.3f}, {L.max():.3f}]{rail}")

    # --- affect circumplex ----------------------------------------------------
    fig1, ax = plt.subplots(figsize=(5, 5))
    for name, aff in MOODS.items():
        ax.scatter(aff.valence, aff.arousal, s=120, zorder=3)
        ax.annotate(name, (aff.valence, aff.arousal), xytext=(6, 4),
                    textcoords="offset points", fontsize=9)
    ax.axhline(0.5, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("valence"); ax.set_ylabel("arousal")
    ax.set_title("Step 1: moods on the affect circumplex")
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.05, 1.05)
    fig1.tight_layout(); fig1.savefig(OUT / "affect_circumplex.png", dpi=130)

    # --- per-mood: cursor path | signals | tendon curves ---------------------
    n = len(MOODS)
    fig2, axes = plt.subplots(n, 3, figsize=(14, 2.1 * n))
    theta = np.linspace(0, 2 * np.pi, 100)
    for i, (name, tr) in enumerate(trajs.items()):
        axp, axs, axl = axes[i]
        axp.plot(np.cos(theta), np.sin(theta), color="gray", lw=0.5)
        axp.plot(tr["cx"], tr["cy"], lw=0.7)
        axp.set_aspect("equal"); axp.set_xlim(-1.1, 1.1); axp.set_ylim(-1.1, 1.1)
        axp.set_title(f"{name}: cursor path", fontsize=9)
        axp.set_xticks([]); axp.set_yticks([])

        m = tr["t"] <= 6.0
        axs.plot(tr["t"][m], tr["cx"][m], label="c_x", lw=0.9)
        axs.plot(tr["t"][m], tr["cy"][m], label="c_y", lw=0.9)
        axs.plot(tr["t"][m], tr["a"][m] / AXIAL_SAFE_M, label="a (norm)", lw=0.9)
        axs.set_ylim(-1.2, 1.2); axs.set_title(f"{name}: cursor+axial", fontsize=9)
        if i == 0:
            axs.legend(fontsize=7, ncol=3, loc="upper right")

        L = tendon_series(tr)
        for j in range(3):
            axl.plot(tr["t"][m], L[m, j], lw=0.9, label=f"tendon {j+1}")
        axl.axhline(ACT_LOW, color="red", lw=0.5, ls=":")
        axl.set_ylim(ACT_LOW - 0.01, ACT_HIGH + 0.01)
        axl.set_title(f"{name}: tendon lengths (m)", fontsize=9)
        if i == 0:
            axl.legend(fontsize=7, ncol=3, loc="upper right")
        if i < n - 1:
            axs.set_xticklabels([]); axl.set_xticklabels([])
        else:
            axs.set_xlabel("time (s)"); axl.set_xlabel("time (s)")
    fig2.tight_layout(); fig2.savefig(OUT / "mood_signals.png", dpi=130)
    print(f"\nsaved plots -> {OUT}/  (affect_circumplex, mood_signals).png")


if __name__ == "__main__":
    main()
