"""What does a 4th servo actually buy? Measure each routing variant's gamut.

DESIGN/RESEARCH (see NEXT_PHASE_EXPLORATION.md). Companion to `tendon_gamut.py`,
which asked the same question about the 2D cursor inside the 3-tendon box. Here
the routing itself is the variable.

Two numbers per variant, kept separate on purpose:

  AFFORDED   sample the raw tendon box directly. This is what the mechanism can
             do, independent of any control scheme -- the ceiling.
  EXPOSED    drive it through the puppet's cursor+channel mapping. This is what
             a human (or the RL policy's action space) can actually reach.

The gap between them is a control-design problem; the AFFORDED column is the
only one that argues for or against buying a servo.

Reported per variant:
  reach      max horizontal tip displacement (m) -- how far it can point
  twist      max accumulated material torsion (deg), see puppet.total_twist_deg
  bend-twist twist produced by pure BEND commands. High = torsion is coupled to
             bending and cannot be commanded independently.
  z-span     tip height range -- axial/droop authority
  vol        occupied fraction of a voxelised tip workspace, vs baseline = 1.0

    .venv/bin/python exploration/dof_study.py            # all variants
    .venv/bin/python exploration/dof_study.py helix5 partial4
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from puppet import Actuation, load_model, measure_rest, total_twist_deg  # noqa: E402

VARIANTS = ["baseline", "straight4", "helix4", "helix5", "helix3",
            "partial4", "twosection"]
N_RANDOM = 200          # raw-box interior samples (on top of every box corner)
N_EXPOSED = 260         # mapping-driven samples; FIXED across variants
SETTLE_STEPS = 1500     # 7.5 s of sim time; the model is well damped
VOXEL = 0.012           # m, workspace occupancy grid


def settle(model, data, ctrl) -> tuple[np.ndarray, float]:
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = ctrl
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_center")
    return data.site_xpos[tip_id].copy(), total_twist_deg(model, data)


def summarise(tips: np.ndarray, twists: np.ndarray) -> dict:
    tips = np.asarray(tips)
    vox = {tuple(v) for v in np.round(tips / VOXEL).astype(int)}
    return {
        "reach": float(np.max(np.hypot(tips[:, 0], tips[:, 1]))),
        "twist": float(np.max(np.abs(twists))),
        "zspan": float(np.ptp(tips[:, 2])),
        "voxels": len(vox),
    }


def afforded(model, data, act: Actuation, rng) -> dict:
    """Sample the raw tendon box -- the mechanism's ceiling, mapping-free.

    Uniform random sampling alone badly underestimates the ceiling: extreme
    poses need COORDINATED pulls, and a random point in a 4-6D box is almost
    never near a corner. So sample the box corners explicitly (that is where
    reach and twist extrema live) and use the random draws to fill the
    interior for the occupancy volume.
    """
    lo, hi = act.low, act.high
    corners = [lo + np.array(bits) * (hi - lo)
               for bits in np.ndindex(*([2] * model.nu))]
    randoms = [lo + rng.random(model.nu) * (hi - lo) for _ in range(N_RANDOM)]
    tips, tw = [], []
    for ctrl in corners + randoms:
        tip, t = settle(model, data, ctrl)
        tips.append(tip)
        tw.append(t)
    return summarise(np.array(tips), np.array(tw))


def exposed(model, data, act: Actuation, rng) -> tuple[dict, float]:
    """Drive through the cursor+channel mapping. Returns (stats, bend_twist).

    Fixed random budget rather than a grid: a grid over (angle, magnitude,
    channels, axial) has a size that depends on how many channels a variant
    has, which would hand extra-DOF variants more samples and inflate their
    occupancy purely as a counting artefact.
    """
    nc = max(len(act.channels), 1)
    tips, tw = [], []
    for _ in range(N_EXPOSED):
        th = rng.uniform(0, 2 * np.pi)
        mag = rng.uniform(0, 0.9)
        ch = rng.uniform(-1, 1, nc) if act.channels else np.zeros(nc)
        a = rng.uniform(-0.04, 0.0)
        tip, t = settle(model, data, act(mag * np.cos(th), mag * np.sin(th), a, ch))
        tips.append(tip)
        tw.append(t)

    # bend-only sweep: twist that comes along for free with pure bending
    bend_only = []
    for th in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        _, t = settle(model, data, act(0.9 * np.cos(th), 0.9 * np.sin(th),
                                       0.0, np.zeros(nc)))
        bend_only.append(t)
    return summarise(np.array(tips), np.array(tw)), float(np.max(np.abs(bend_only)))


def main() -> None:
    names = sys.argv[1:] or VARIANTS
    rng = np.random.default_rng(7)
    rows = []
    t0 = time.perf_counter()
    for name in names:
        model, meta = load_model(name)
        act = Actuation(meta, measure_rest(model))
        data = mujoco.MjData(model)
        aff = afforded(model, data, act, rng)
        exp, bend_tw = exposed(model, data, act, rng)
        rows.append((name, model.nu, aff, exp, bend_tw,
                     [c.name for c in act.channels]))
        print(f"  ... {name} done ({time.perf_counter() - t0:.0f}s)")

    base_aff = rows[0][2]["voxels"] if rows else 1
    base_exp = rows[0][3]["voxels"] if rows else 1

    print("\n  AFFORDED -- raw tendon box (the mechanism's ceiling)")
    print(f"  {'variant':<11} {'n':>2} {'reach m':>8} {'twist':>7} {'z-span':>7} {'vol':>6}")
    for n, nu, a, _, _, _ in rows:
        print(f"  {n:<11} {nu:2d} {a['reach']:8.4f} {a['twist']:6.1f}d "
              f"{a['zspan']:7.4f} {a['voxels'] / base_aff:6.2f}")

    print("\n  EXPOSED -- through the cursor + channel mapping")
    print(f"  {'variant':<11} {'n':>2} {'reach m':>8} {'twist':>7} {'bend-tw':>8} "
          f"{'z-span':>7} {'vol':>6}  channels")
    for n, nu, _, e, bt, ch in rows:
        print(f"  {n:<11} {nu:2d} {e['reach']:8.4f} {e['twist']:6.1f}d "
              f"{bt:7.1f}d {e['zspan']:7.4f} {e['voxels'] / base_exp:6.2f}  "
              f"{', '.join(ch) or '-'}")
    print(f"\n  ({time.perf_counter() - t0:.0f}s total; voxel {VOXEL * 1000:.0f} mm, "
          f"{N_RANDOM} raw samples/variant)")


if __name__ == "__main__":
    main()
