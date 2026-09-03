"""Parametric tentacle model generator -- explore alternative tendon routings.

DESIGN/RESEARCH (see NEXT_PHASE_EXPLORATION.md). A fork of the POC's
`training/rl/generate_mujoco_xml.py` with the tendon routing opened up as
parameters, so we can ask "what would a 4th servo actually buy us?" by
*driving* each variant instead of arguing about it.

The POC routes 3 tendons straight up the body at fixed angles (60/180/300 deg),
all running the full length. Two structural consequences:

  * a straight tendon has zero moment arm about the backbone axis, so it can
    never produce TWIST -- only bend + axial. A 4th straight cable adds an
    actuator but no new shape (just co-contraction / stiffness modulation).
  * every tendon spans the whole body, so the tentacle is a single
    constant-curvature section: it can make a C, never an S.

So this generator exposes the two knobs that break those limits:

  helix_deg   total angular wrap of a tendon over the body length. Non-zero =>
              the cable spirals => tension gains a moment arm about the axis
              => real torsion authority. (Mechanically: helical channels, or
              indexing each printed vertebra by a constant angle at assembly.)
  t_end       fraction of body length at which the tendon terminates. Tendons
              ending early only bend the PROXIMAL part => a second independent
              section => S-curves.

Geometry (joint positions, anchor heights, stiffness taper, radii) is copied
verbatim from the POC generator so variants stay physically comparable.

    # the POC model, regenerated through this code path (sanity baseline)
    .venv/bin/python exploration/tentacle_variants.py baseline

    # 3 straight + 1 counter-helical 4th cable  -> twist channel
    .venv/bin/python exploration/tentacle_variants.py helix4

    # 3 full-length + 1 short cable ending at 55% -> partial 2nd section
    .venv/bin/python exploration/tentacle_variants.py partial4

    # all presets
    .venv/bin/python exploration/tentacle_variants.py all

Each variant writes `exploration/assets/<name>.xml` plus a `<name>.json`
sidecar describing the tendons (angle / helix / t_end / rest length), which
`puppet.py` reads to build the right cursor->tendon mapping.
"""

from __future__ import annotations

import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from xml.dom import minidom

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MESHDIR = REPO / "assets" / "simulation" / "mujoco_assets"
OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(exist_ok=True)

# --- geometry copied from the POC generator (keep variants comparable) -------
ROTATION_OFFSET_DEG = 60.0
ACTUATOR_KP = 200.0
ACTUATOR_FORCERANGE = "-200 0"
# The POC declares ctrlrange [0.12, 0.34] while the cable's geometric rest
# length is 0.2262 m (measured: actuator_force hits exactly 0.0 above 0.2262).
# The range above rest is NOT dead -- it is antagonist release. At the straight
# pose those commands do nothing, but once the tentacle bends, an off-side
# cable's actual path length grows past its command and it starts pulling back;
# commanding it longer lets it go slack. Both halves matter, so variants keep
# the same stroke below rest and the same release headroom above it.
POC_REST_LENGTH = 0.2262
POC_CTRL_LOW = 0.12
POC_CTRL_HIGH = 0.34
SHORTEN_FRAC = 1.0 - POC_CTRL_LOW / POC_REST_LENGTH          # ~0.4695
RELEASE_RATIO = POC_CTRL_HIGH / POC_REST_LENGTH              # ~1.5031
JOINT_RANGE = "0 0.9"
E = 2.0e7
BASE_D, BASE_T = 3.4e-3, 0.93e-3
TIP_D, TIP_T = 1.5e-3, 0.50e-3
DAMPING_FACTOR = 0.07

JOINT_Z = [
    14.3255e-3, 34.275e-3, 52.814e-3, 70.0425e-3, 86.0535e-3, 100.9325e-3,
    114.76e-3, 127.61e-3, 139.55e-3, 150.649e-3, 160.962e-3, 170.55e-3,
    179.453e-3, 187.73e-3, 195.42e-3, 202.571e-3, 209.2135e-3, 215.3875e-3,
    221.1245e-3, 226.4565e-3,
]
ANCHOR_Z = [
    (5e-3, 12.02e-3), (20.421e-3, 32.131e-3), (39.939e-3, 50.821e-3),
    (58.077e-3, 68.19e-3), (74.932e-3, 84.331e-3), (90.597e-3, 99.331e-3),
    (105.154e-3, 113.271e-3), (118.682e-3, 126.225e-3), (131.254e-3, 138.264e-3),
    (142.938e-3, 149.452e-3), (153.795e-3, 159.849e-3), (163.885e-3, 169.511e-3),
    (173.262e-3, 178.49e-3), (181.976e-3, 186.835e-3), (190.074e-3, 194.59e-3),
    (197.60e-3, 201.79e-3), (204.594e-3, 208.493e-3), (211.093e-3, 214.717e-3),
    (217.133e-3, 220.501e-3), (222.746e-3, 225.876e-3), (227.963e-3, 230.871e-3),
]
R_START, R_END = 13.74e-3, 2.35e-3          # tendon radius, base -> tip

Z0, Z1 = ANCHOR_Z[0][0], ANCHOR_Z[-1][1]
DZ = Z1 - Z0
NUM_BODIES = len(JOINT_Z) + 1


@dataclass
class Tendon:
    """One cable's routing.

    angle_deg: angular position at the BASE, in the body frame.
    helix_deg: total angular wrap accumulated from base to t_end. 0 = straight.
               Sign sets handedness -> sign of the torsion it produces.
    t_end:     fraction of body length where the cable anchors (1.0 = tip).
    """

    angle_deg: float
    helix_deg: float = 0.0
    t_end: float = 1.0


def joint_stiffness(i: int, n: int) -> tuple[float, float]:
    """Bending stiffness/damping for segment i, tapered like the POC model."""
    if n <= 1:
        return 0.3, 0.01
    alpha = i / float(n - 1)
    d_i = BASE_D * (TIP_D / BASE_D) ** alpha
    t_i = BASE_T * (TIP_T / BASE_T) ** alpha
    I_i = (math.pi * (d_i ** 4)) / 64.0
    K_i = (E * I_i) / t_i
    return K_i, DAMPING_FACTOR * K_i


def route_xyz(td: Tendon, t: float) -> tuple[float, float, float]:
    """Global 3D point on tendon `td` at body-length fraction t in [0, 1].

    Straight routing (helix_deg=0) reproduces the POC exactly: z and radius
    interpolate linearly, angle is constant. A non-zero helix advances the
    angle with height -- which is the whole point, since that is what gives
    the cable a moment arm about the backbone axis.
    """
    z = Z0 + t * DZ
    r = R_START + t * (R_END - R_START)
    ang = math.radians(td.angle_deg) + math.radians(td.helix_deg) * t
    return r * math.cos(ang), r * math.sin(ang), z


def rest_length(td: Tendon) -> float:
    """Polyline length of the cable at rest.

    At qpos=0 every body frame coincides with the world frame, so the local
    site coordinates we emit ARE the rest-pose global coordinates -- summing
    the segments gives the true rest length. Needed because helical and
    partial cables are not 0.34 m, so their ctrlrange must be scaled.
    """
    pts = [route_xyz(td, t) for t in np.linspace(0.0, td.t_end, 200)]
    return float(sum(math.dist(a, b) for a, b in zip(pts, pts[1:])))


def build(name: str, tendons: list[Tendon]) -> tuple[str, dict]:
    """Emit the MuJoCo XML + a sidecar dict describing the actuation."""
    root = ET.Element("mujoco", attrib={"model": name})
    # absolute meshdir so variants can live outside assets/simulation/
    ET.SubElement(root, "compiler", attrib={
        "angle": "radian", "autolimits": "true", "meshdir": str(MESHDIR)})
    ET.SubElement(root, "option", attrib={"timestep": "0.005", "iterations": "50"})

    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", attrib={
        "stiffness": "0.0", "damping": "0.0", "range": JOINT_RANGE})
    ET.SubElement(default, "geom", attrib={
        "type": "mesh", "xyaxes": "0 0 -1 -1 0 0", "contype": "0",
        "conaffinity": "0", "margin": "0.001", "solref": "0.01 1",
        "solimp": "0.9 0.95 0.001"})

    asset = ET.SubElement(root, "asset")
    for i in range(1, NUM_BODIES + 1):
        ET.SubElement(asset, "mesh", attrib={
            "name": f"mesh{i}", "file": f"Part{i}.stl",
            "scale": "0.001 0.001 0.001"})
    ET.SubElement(asset, "texture", attrib={
        "type": "skybox", "builtin": "gradient", "rgb1": "0.3 0.5 0.7",
        "rgb2": "0 0 0", "width": "512", "height": "3072"})
    ET.SubElement(asset, "texture", attrib={
        "type": "2d", "name": "groundplane", "builtin": "checker", "mark": "edge",
        "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3", "markrgb": "0.8 0.8 0.8",
        "width": "300", "height": "300"})
    ET.SubElement(asset, "material", attrib={
        "name": "groundplane", "texture": "groundplane", "texuniform": "true",
        "texrepeat": "5 5", "reflectance": "0.2"})

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", attrib={
        "diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"})
    ET.SubElement(visual, "rgba", attrib={"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(visual, "global", attrib={
        "azimuth": "150", "elevation": "-20",
        "offwidth": "1280", "offheight": "960"})   # headroom for offscreen renders

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", attrib={
        "pos": "0 0 3", "dir": "0 0 -1", "directional": "false"})
    ET.SubElement(worldbody, "geom", attrib={
        "name": "floor", "pos": "0 0 -0.2", "size": "0 0 .125", "type": "plane",
        "material": "groundplane", "conaffinity": "0", "condim": "3",
        "xyaxes": "1 0 0 0 1 0"})
    ET.SubElement(worldbody, "camera", attrib={
        "name": "fixed_overview", "mode": "fixed", "pos": "0 -0.5 0.7",
        "quat": "0.924 0.383 0 0", "fovy": "60"})
    ET.SubElement(worldbody, "site", attrib={
        "name": "target", "type": "sphere", "size": "0.01 0.01 0.01",
        "rgba": "1 0 0 0.8", "pos": "0 0 0"})

    # --- body chain ---------------------------------------------------------
    bodies = []
    body = ET.SubElement(worldbody, "body", attrib={"name": "body1", "pos": "0 0 0"})
    ET.SubElement(body, "geom", attrib={
        "name": "geom1", "type": "mesh", "mesh": "mesh1",
        "contype": "0", "conaffinity": "0"})
    bodies.append(body)
    for i, z in enumerate(JOINT_Z):
        K_i, D_i = joint_stiffness(i, len(JOINT_Z))
        nb = ET.SubElement(body, "body", attrib={"name": f"body{i+2}", "pos": "0 0 0"})
        ET.SubElement(nb, "geom", attrib={
            "name": f"geom{i+2}", "type": "mesh", "mesh": f"mesh{i+2}",
            "contype": "0", "conaffinity": "0"})
        ET.SubElement(nb, "joint", attrib={
            "type": "ball", "name": f"joint{i+1}", "pos": f"0 0 {z}",
            "stiffness": f"{K_i:.5g}", "damping": f"{D_i:.5g}"})
        bodies.append(nb)
        body = nb

    # --- tendon sites -------------------------------------------------------
    site_names: list[list[str]] = [[] for _ in tendons]
    for i, (z_in, z_out) in enumerate(ANCHOR_Z):
        t_in = (z_in - Z0) / DZ
        t_out = (z_out - Z0) / DZ
        for k, td in enumerate(tendons):
            for tag, t in (("in", t_in), ("out", t_out)):
                if t > td.t_end + 1e-9:
                    continue                      # cable already terminated
                x, y, z = route_xyz(td, t)
                nm = f"site_{tag}_{i}_{k}"
                ET.SubElement(bodies[i], "site", attrib={
                    "name": nm, "pos": f"{x:.5f} {y:.5f} {z:.5f}",
                    "size": "0.001", "rgba": "1 1 0 1"})
                site_names[k].append(nm)

    ET.SubElement(bodies[-1], "site", attrib={
        "name": "tip_center", "pos": f"0 0 {ANCHOR_Z[-1][1]}",
        "size": "0.002", "rgba": "0 1 0 1"})
    # off-axis tip marker: lets us read TWIST off the tip frame directly
    ET.SubElement(bodies[-1], "site", attrib={
        "name": "tip_roll_ref", "pos": f"0.004 0 {ANCHOR_Z[-1][1]}",
        "size": "0.0015", "rgba": "0 0.6 1 1"})

    # --- tendons + actuators ------------------------------------------------
    tendon_elem = ET.SubElement(root, "tendon")
    for k, names in enumerate(site_names):
        sp = ET.SubElement(tendon_elem, "spatial", attrib={
            "name": f"tendon_{k+1}", "width": "0.001", "rgba": "1 0 0 1"})
        for nm in names:
            ET.SubElement(sp, "site", attrib={"site": nm})

    act = ET.SubElement(root, "actuator")
    rests = []
    for k, td in enumerate(tendons):
        rest = rest_length(td)
        rests.append(rest)
        lo = rest * (1.0 - SHORTEN_FRAC)      # same fractional stroke as the POC
        hi = rest * RELEASE_RATIO             # ... and the same release headroom
        ET.SubElement(act, "position", attrib={
            "name": f"actuator_{k+1}", "tendon": f"tendon_{k+1}",
            "kp": str(ACTUATOR_KP), "forcerange": ACTUATOR_FORCERANGE,
            "ctrlrange": f"{lo:.4f} {hi:.4f}"})

    sensor = ET.SubElement(root, "sensor")
    for k in range(len(tendons)):
        ET.SubElement(sensor, "tendonpos", attrib={
            "name": f"tendon{k+1}_pos", "tendon": f"tendon_{k+1}"})
    ET.SubElement(sensor, "framepos", attrib={
        "name": "tip_pos", "objtype": "site", "objname": "tip_center"})
    ET.SubElement(sensor, "framequat", attrib={
        "name": "tip_quat", "objtype": "site", "objname": "tip_center"})

    xml = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(indent="  ")
    meta = {
        "name": name,
        "tendons": [asdict(t) for t in tendons],
        "rest_lengths": rests,
        "ctrl_low": [r * (1.0 - SHORTEN_FRAC) for r in rests],
        "ctrl_high": [r * RELEASE_RATIO for r in rests],
    }
    return xml, meta


# --- presets ----------------------------------------------------------------
# Base angles match the POC (60/180/300 deg).
_A = [ROTATION_OFFSET_DEG + 120.0 * k for k in range(3)]

PRESETS: dict[str, list[Tendon]] = {
    # 1. the POC, regenerated here. Sanity baseline -- should behave identically.
    "baseline": [Tendon(a) for a in _A],

    # 2. 4th STRAIGHT cable at 90 deg spacing. The "obvious" 4th servo.
    #    Prediction: no new shape, only co-contraction / stiffness.
    "straight4": [Tendon(0.0), Tendon(90.0), Tendon(180.0), Tendon(270.0)],

    # 3. 3 straight + 1 helical 4th cable (one full 180 deg wrap).
    #    The cheapest real twist channel: 3 servos keep the POC bend mapping,
    #    the 4th adds torsion (coupled to some bend -- that's the tradeoff).
    "helix4": [Tendon(a) for a in _A] + [Tendon(0.0, helix_deg=180.0)],

    # 4. all three cables helical, same handedness. Bend still spans 2D, but
    #    every bend now drags twist with it -- a "writhe" character.
    "helix3": [Tendon(a, helix_deg=120.0) for a in _A],

    # 5. counter-wound pair added to the 3 straight cables (5 servos). Gives
    #    twist in BOTH directions -- but one cable at a time, so each direction
    #    drags its own bend along. Measured: chan=+1 -> -63 deg twist AND
    #    0.064 m of reach. Not the clean twist axis it looks like.
    "helix5": [Tendon(a) for a in _A]
              + [Tendon(0.0, helix_deg=180.0), Tendon(180.0, helix_deg=-180.0)],

    # 6. the actually-clean twist axis: two helical cables of the SAME
    #    handedness at OPPOSITE base angles. Their bend contributions oppose
    #    and cancel; their torsions share a sign and add. (Counter-winding, as
    #    in helix5, cancels the torsion too -- which is the trap.) Cost: twist
    #    in one direction only, relying on the elastomer to spring back.
    "twistpair": [Tendon(a) for a in _A]
                 + [Tendon(0.0, helix_deg=180.0), Tendon(180.0, helix_deg=180.0)],

    # 7. 3 full-length + 1 cable anchoring at 55% height.
    #    Breaks the single-arc limit in ONE plane -> asymmetric S-curves.
    "partial4": [Tendon(a) for a in _A] + [Tendon(_A[0] + 60.0, t_end=0.55)],

    # 8. two full 3-cable sections (6 servos) -- the "real" S-curve robot,
    #    for reference on what the expressive ceiling looks like.
    "twosection": [Tendon(a, t_end=0.55) for a in _A]
                  + [Tendon(a + 60.0) for a in _A],
}


def write(name: str) -> None:
    xml, meta = build(name, PRESETS[name])
    (OUT / f"{name}.xml").write_text(xml)
    (OUT / f"{name}.json").write_text(json.dumps(meta, indent=2))
    rests = ", ".join(f"{r:.3f}" for r in meta["rest_lengths"])
    print(f"  {name:<11} {len(PRESETS[name])} tendons   rest lengths: [{rests}] m")


def main() -> None:
    args = sys.argv[1:] or ["all"]
    names = list(PRESETS) if args == ["all"] else args
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        sys.exit(f"unknown preset(s) {unknown}; available: {list(PRESETS)}")
    print(f"writing to {OUT}")
    for n in names:
        write(n)


if __name__ == "__main__":
    main()
