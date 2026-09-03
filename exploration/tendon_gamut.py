"""Reachability study: what does the 2D cursor leave unreachable in 3-tendon space?

DESIGN/RESEARCH — imports the POC's real 2D->tendon mapping read-only; does NOT
modify any POC code. See NEXT_PHASE_EXPLORATION.md.

The 3 tendon lengths form a 3D control box C = [0.12, 0.34]^3 (from the MuJoCo
ctrlrange). The 2D cursor is a function f: disk(|c|<=1) -> C, so its image M is a
2D *sheet* folded inside a 3D *volume*. Everything in C off the sheet is motion
2D physically cannot command. This script measures that gap three ways:
  1. tolerance sweep  -- fraction of C within eps of the 2D sheet, vs eps
  2. nearest-distance -- how far the unreachable regions sit
  3. named probes     -- distance of specific expressive moves from the sheet

Run headless (saves PNGs to exploration/out/):
    .venv/bin/python exploration/tendon_gamut.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# shoggoth_mini resolves via cwd; ensure the repo root
# (which contains the shoggoth_mini/ package) is importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- read-only import of the REAL POC mapping + geometry constants -------------
from shoggoth_mini.control.geometry import convert_2d_cursor_to_target_lengths
from shoggoth_mini.common.constants import MOTOR_NORMALIZED_POSITIONS, MOTOR_NAMES

# --- ground truth (from configs + generate_mujoco_xml, verified) --------------
ACT_LOW = 0.12          # ctrlrange min  (rest 0.34 - max_shortening 0.22)
ACT_HIGH = 0.34         # ctrlrange max  (= tendon rest length)
BASELINE = 0.23         # initial_actuator_position default (nominal rest pose)
MAX_2D_MAG = 1.0        # max_2d_action_magnitude default

baseline_lengths = np.full(3, BASELINE, dtype=np.float32)
act_low = np.full(3, ACT_LOW, dtype=np.float32)
act_high = np.full(3, ACT_HIGH, dtype=np.float32)

CUBE_SIDE = ACT_HIGH - ACT_LOW              # 0.22 m
CUBE_DIAG = CUBE_SIDE * np.sqrt(3)          # ~0.381 m
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)


def cursor_to_lengths(cursor_2d: np.ndarray) -> np.ndarray:
    """Wrap the POC function (which takes one cursor) for a single 2-vector."""
    return convert_2d_cursor_to_target_lengths(
        cursor_2d, baseline_lengths, act_low, act_high, MAX_2D_MAG
    )


def sample_2d_sheet(n_r: int = 40, n_theta: int = 120) -> np.ndarray:
    """Sample the cursor disk in polar coords, push through the real mapping.

    Returns (N, 3) cloud of tendon-length configs reachable by 2D control.
    Polar because the map is direction (theta = bend dir) + magnitude (r = amount).
    """
    pts = [cursor_to_lengths(np.array([0.0, 0.0]))]  # r=0 -> baseline
    radii = np.linspace(0.0, MAX_2D_MAG, n_r + 1)[1:]
    thetas = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    for r in radii:
        for th in thetas:
            c = np.array([r * np.cos(th), r * np.sin(th)], dtype=np.float32)
            pts.append(cursor_to_lengths(c))
    return np.unique(np.round(np.array(pts), 6), axis=0)


def sample_cube(n: int = 34) -> np.ndarray:
    """Grid the full 3-tendon box C = [ACT_LOW, ACT_HIGH]^3."""
    ax = np.linspace(ACT_LOW, ACT_HIGH, n)
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


def nearest_dist(query: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    """Distance from each query point to the nearest point in cloud."""
    try:
        from scipy.spatial import cKDTree

        d, _ = cKDTree(cloud).query(query)
        return d
    except Exception:
        # brute-force fallback, chunked to bound memory
        out = np.empty(len(query))
        for i in range(0, len(query), 2000):
            q = query[i : i + 2000]
            d = np.sqrt(((q[:, None, :] - cloud[None, :, :]) ** 2).sum(-1))
            out[i : i + 2000] = d.min(1)
        return out


def main() -> None:
    print("=" * 70)
    print("2D-cursor reachability inside the 3-tendon control box")
    print("=" * 70)
    print(f"  box C = [{ACT_LOW}, {ACT_HIGH}]^3   side={CUBE_SIDE:.3f} m   "
          f"diag={CUBE_DIAG*1000:.1f} mm")
    print(f"  tendon dirs (120 deg): "
          f"{[MOTOR_NORMALIZED_POSITIONS[m].tolist() for m in MOTOR_NAMES]}")
    print(f"  sum of tendon dirs = "
          f"{sum(MOTOR_NORMALIZED_POSITIONS[m] for m in MOTOR_NAMES).round(3).tolist()}"
          "  -> zero => 2D cannot command common-mode (all-shorten) motion\n")

    sheet = sample_2d_sheet()
    cube = sample_cube()
    print(f"  sampled 2D sheet: {len(sheet):5d} pts   cube grid: {len(cube):5d} pts\n")

    dist = nearest_dist(cube, sheet)         # metres, per cube point
    dist_mm = dist * 1000.0

    # --- 1. tolerance sweep ---------------------------------------------------
    print("TOLERANCE SWEEP  (fraction of the 3-tendon box within eps of 2D sheet)")
    print(f"  {'eps (mm)':>9} | {'% of box':>8} | {'% of diag':>9}")
    print("  " + "-" * 32)
    for eps_mm in [1, 2, 5, 10, 15, 20, 30, 40]:
        frac = float((dist_mm <= eps_mm).mean()) * 100.0
        print(f"  {eps_mm:9d} | {frac:7.1f}% | {eps_mm/(CUBE_DIAG*1000)*100:8.1f}%")
    print(f"\n  max distance any box point sits from the sheet: "
          f"{dist_mm.max():.1f} mm ({dist_mm.max()/(CUBE_DIAG*1000)*100:.0f}% of diag)")
    print(f"  median distance: {np.median(dist_mm):.1f} mm\n")

    # --- 3. named expressive probes -------------------------------------------
    b = BASELINE
    probes = {
        "baseline (rest)          [on-sheet sanity]": np.array([b, b, b]),
        "pure bend -> motor 1     [on-sheet sanity]": cursor_to_lengths(np.array([0.0, 1.0])),
        "axial contract (recoil)  [-0.08 all]":       np.array([b-0.08, b-0.08, b-0.08]),
        "deep axial contract      [-0.11 all]":       np.array([b-0.11, b-0.11, b-0.11]),
        "axial extend (droop)     [+0.08 all]":       np.array([b+0.08, b+0.08, b+0.08]),
        "asym twist               [.15 .20 .28]":     np.array([0.15, 0.20, 0.28]),
    }
    print("NAMED PROBES  (distance of specific moves from the 2D sheet)")
    print(f"  {'move':<44} | {'dist mm':>7} | {'% diag':>6} | reachable(<=5mm)?")
    print("  " + "-" * 78)
    for name, cfg in probes.items():
        d_mm = float(nearest_dist(cfg[None, :], sheet)[0]) * 1000.0
        tag = "YES" if d_mm <= 5 else "no"
        print(f"  {name:<44} | {d_mm:7.1f} | {d_mm/(CUBE_DIAG*1000)*100:5.1f}% | {tag}")
    print()

    # --- 4. HYBRID: 2D sheet + one axial channel a*[1,1,1] --------------------
    # Extrude the sheet along its normal [1,1,1] -> a (clipped, warped) cylinder.
    axis = np.ones(3) / np.sqrt(3.0)  # unit [1,1,1]

    def build_hybrid(delta_m: float, n_a: int = 41) -> np.ndarray:
        """Sweep axial offset a in [-delta, +delta], clip to the box."""
        offs = np.linspace(-delta_m, delta_m, n_a)
        chunks = [np.clip(sheet + a * axis, ACT_LOW, ACT_HIGH) for a in offs]
        return np.unique(np.round(np.vstack(chunks), 6), axis=0)

    print("HYBRID (2D + axial channel): coverage vs pure 2D")
    print(f"  {'eps (mm)':>9} | {'2D only':>8} | {'+axial ±80mm':>12} | {'+axial box-max':>14}")
    print("  " + "-" * 54)
    hyb_safe = build_hybrid(0.08)     # realistic, matches recoil/droop magnitude
    hyb_max = build_hybrid(0.22)      # box-limited ceiling (clip handles overflow)
    d_safe = nearest_dist(cube, hyb_safe) * 1000.0
    d_max = nearest_dist(cube, hyb_max) * 1000.0
    for eps_mm in [1, 2, 5, 10, 15, 20, 30, 40]:
        f2d = (dist_mm <= eps_mm).mean() * 100
        fs = (d_safe <= eps_mm).mean() * 100
        fm = (d_max <= eps_mm).mean() * 100
        print(f"  {eps_mm:9d} | {f2d:7.1f}% | {fs:11.1f}% | {fm:13.1f}%")
    print(f"\n  sheet pts: {len(sheet)}   hybrid(±80mm): {len(hyb_safe)}   "
          f"hybrid(box-max): {len(hyb_max)}")

    print("\n  probe distances: pure-2D -> 2D+axial(±80mm)")
    print(f"  {'move':<30} | {'2D mm':>7} | {'+axial mm':>9} | closed?")
    print("  " + "-" * 62)
    for name, cfg in probes.items():
        if "sanity" in name:
            continue
        d2 = float(nearest_dist(cfg[None, :], sheet)[0]) * 1000.0
        dh = float(nearest_dist(cfg[None, :], hyb_safe)[0]) * 1000.0
        tag = "YES" if dh <= 5 else ("residual" if dh < d2 * 0.7 else "no")
        short = name.split("[")[0].strip()
        print(f"  {short:<30} | {d2:7.1f} | {dh:8.1f} | {tag}")
    print()

    # --- plots ----------------------------------------------------------------
    # (a) sheet folded inside the box
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sheet[:, 0], sheet[:, 1], sheet[:, 2], s=2, alpha=0.5,
               c="tab:blue", label="2D-reachable sheet")
    for name, cfg in probes.items():
        if "sanity" in name:
            continue
        ax.scatter(*cfg, s=60, marker="X",
                   label=name.split("[")[0].strip())
    r = [ACT_LOW, ACT_HIGH]
    for s in r:
        for t in r:
            ax.plot([r[0], r[1]], [s, s], [t, t], c="gray", lw=0.4, alpha=0.5)
            ax.plot([s, s], [r[0], r[1]], [t, t], c="gray", lw=0.4, alpha=0.5)
            ax.plot([s, s], [t, t], [r[0], r[1]], c="gray", lw=0.4, alpha=0.5)
    ax.set_xlabel("tendon 1 (m)"); ax.set_ylabel("tendon 2 (m)"); ax.set_zlabel("tendon 3 (m)")
    ax.set_title("2D cursor = a 2D sheet inside the 3D tendon box")
    ax.legend(loc="upper left", fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / "sheet_in_box.png", dpi=130)

    # (a2) hybrid cylinder (2D sheet extruded along [1,1,1])
    figc = plt.figure(figsize=(7, 6))
    axc = figc.add_subplot(111, projection="3d")
    axc.scatter(hyb_safe[:, 0], hyb_safe[:, 1], hyb_safe[:, 2], s=1, alpha=0.15,
                c="tab:green", label="2D + axial (±80mm)")
    axc.scatter(sheet[:, 0], sheet[:, 1], sheet[:, 2], s=2, alpha=0.5,
                c="tab:blue", label="2D sheet")
    for name, cfg in probes.items():
        if "sanity" in name:
            continue
        axc.scatter(*cfg, s=60, marker="X", label=name.split("[")[0].strip())
    axc.set_xlabel("tendon 1 (m)"); axc.set_ylabel("tendon 2 (m)"); axc.set_zlabel("tendon 3 (m)")
    axc.set_title("2D + axial channel = sheet extruded into a cylinder")
    axc.legend(loc="upper left", fontsize=6)
    figc.tight_layout(); figc.savefig(OUT / "hybrid_cylinder.png", dpi=130)

    # (b) tolerance curve
    eps_grid = np.linspace(0, 50, 200)
    cov = [(dist_mm <= e).mean() * 100 for e in eps_grid]
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(eps_grid, cov, lw=2)
    ax2.set_xlabel("tolerance eps (mm)"); ax2.set_ylabel("% of 3-tendon box reachable")
    ax2.set_title("How much of the box 2D reaches, allowing slop")
    ax2.grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig(OUT / "tolerance_curve.png", dpi=130)

    # (c) distance histogram
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.hist(dist_mm, bins=50, color="tab:purple", alpha=0.8)
    ax3.set_xlabel("distance to 2D sheet (mm)"); ax3.set_ylabel("# box configs")
    ax3.set_title("Distribution of unreachability across the box")
    fig3.tight_layout(); fig3.savefig(OUT / "distance_hist.png", dpi=130)

    print(f"saved plots -> {OUT}/  (sheet_in_box, tolerance_curve, distance_hist).png")


if __name__ == "__main__":
    main()
