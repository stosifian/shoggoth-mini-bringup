"""Hand-puppet the tentacle in MuJoCo -- trackpad = 2D bend, keys/scroll = axial.

DESIGN/RESEARCH (see NEXT_PHASE_EXPLORATION.md). Read-only w.r.t. the POC.

The point is INTUITION: what shapes does this tendon architecture actually
afford? `tendon_gamut.py` answered that statistically (the 2D cursor is a
2-manifold reaching ~12% of the tendon box); this lets you feel it. Drive the
cursor by hand, watch the tip trace fill in, watch cables go slack, watch the
thing buckle.

    mjpython exploration/puppet.py                  # POC-equivalent 3-tendon
    mjpython exploration/puppet.py partial4         # + a cable ending at 55%
    mjpython exploration/puppet.py helix5           # + counter-wound twist pair
    mjpython exploration/puppet.py poc              # the literal POC XML

    mjpython exploration/puppet.py --play           # replay the newest take
    mjpython exploration/puppet.py --play NAME.npz --loop

macOS needs `mjpython` (the viewer wants the main thread) and Accessibility
permission for the terminal, so pynput can read the cursor -- same permission
the POC's trackpad_control uses (`tools/check_input_permissions.py`).

Three things worth knowing before you drive it, all measured, not assumed:

  * cable rest length is 0.2262 m, not the 0.34 m the POC's ctrlrange implies.
    On each bar, the yellow tick is rest: below it the cable pulls, above it
    the cable is being PAID OUT. At the straight pose paying out does nothing,
    but a bent tentacle stretches its off-side cables past their command, so
    they fight the motion unless released -- that release is most of what makes
    the direction you command track the direction you get.
  * common-mode pull does not telescope the tentacle, it BUCKLES it: uniform
    0.12 takes tip height 0.231 -> 0.042 m while tip radius grows to 0.065 m,
    in a direction physics picks, not you. Ride `a` down slowly and watch for
    where it stops being a droop and starts being a collapse.
  * co-contraction is a real shape channel, not just stiffness. Holding the
    antagonists at rest instead of releasing them ADDS reach (0.063 -> 0.074 m
    at one working point) because their tension compresses the body and
    amplifies the existing bend.

Two drive modes, toggled with `m`:
    CURSOR  the trackpad steers the bend ring through the POC's cursor->cable
            mapping -- one 2D command, all cables move together. This is the
            action space the RL policy and the primitives actually speak, so
            it is the honest answer to "what can the robot be told to do".
    RAW     the trackpad is ignored and you set ONE CABLE AT A TIME (select
            with 1..9, adjust with , / .). No mapping in the way. Use it to
            build the underlying intuition -- what does cable 2 alone do, where
            does a cable go slack, what does the mapping refuse to ask for.

Controls (focus the MuJoCo window for keys; the trackpad works anywhere):
    trackpad / mouse   bend cursor (c_x, c_y) -- screen centre = neutral
    UP / DOWN arrows   axial common mode `a`  (DOWN = pull = shorten all)
    scroll             same axial channel
    SPACE              freeze/unfreeze cursor input (park a pose and inspect)
    0                  recentre axial to 0
    m                  toggle CURSOR <-> RAW (see above)
    1..9               RAW: select cable       CURSOR: select extra channel
    , / .              adjust the selected cable / channel
    /                  reset it (channel to 0, or all cables back to rest)
    r                  start/stop recording (auto-saves on stop)
    ENTER              replay the loaded take (the last one recorded, or
                       whatever --play named). ENTER again cancels a running
                       replay. The recorded tip path is drawn in orange as a
                       ghost, the live one in green on top: they should sit on
                       each other, and where they do not is a state the replay
                       did not reproduce.
    L                  toggle looping replay
    t                  trace on/off      c   clear trace + ghost
    p                  probe: sweep the cursor in a circle, report the
                       commanded-azimuth -> actual-tip-azimuth mapping
    q                  quit
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ASSETS = HERE / "assets"
GESTURES = HERE / "out" / "gestures"
GESTURES.mkdir(parents=True, exist_ok=True)

AXIAL_STEP = 0.002          # m per scroll click / keypress
AXIAL_LIMIT = 0.11          # m; full stroke is ~0.106, so this is "everything"
TRACE_MAX = 400
KEY_UP, KEY_DOWN = 265, 264             # GLFW_KEY_UP / GLFW_KEY_DOWN
KEY_ENTER, KEY_KP_ENTER = 257, 335      # GLFW_KEY_ENTER / _KP_ENTER
GHOST_MAX = 150             # ghost path segments; the scene geom budget is finite
CHAN_STEP = 0.1             # per keypress on an extra channel: 10 taps end-to-end
POC_ANGLES_DEG = [90.0, -30.0, 210.0]   # MOTOR_NORMALIZED_POSITIONS, as angles


# =============================================================================
# model + actuation description
# =============================================================================
def load_model(which: str):
    """Return (model, meta). `meta` describes each tendon's routing + limits."""
    if which == "poc":
        path = REPO / "assets" / "simulation" / "tentacle.xml"
        model = mujoco.MjModel.from_xml_path(str(path))
        # the POC XML carries no sidecar; reconstruct it from the control
        # convention in common/constants.py + the declared ctrlrange
        meta = {
            "name": "poc",
            "tendons": [{"angle_deg": a, "helix_deg": 0.0, "t_end": 1.0}
                        for a in POC_ANGLES_DEG],
            "ctrl_low": list(model.actuator_ctrlrange[:, 0]),
            "ctrl_high": list(model.actuator_ctrlrange[:, 1]),
        }
        return model, meta

    xml, side = ASSETS / f"{which}.xml", ASSETS / f"{which}.json"
    if not xml.exists():
        sys.exit(f"no such variant '{which}'. Generate one first:\n"
                 f"  .venv/bin/python exploration/tentacle_variants.py all\n"
                 f"available: {sorted(p.stem for p in ASSETS.glob('*.xml'))} + 'poc'")
    return mujoco.MjModel.from_xml_path(str(xml)), json.loads(side.read_text())


def measure_rest(model) -> np.ndarray:
    """Cable lengths with every actuator fully slack -- the true neutral pose.

    Measured rather than read off ctrlrange: the POC's declared upper bound
    (0.34 m) sits well above the geometric rest length (0.2262 m), so using it
    as the neutral would put the whole mapping 0.11 m into slack. The range
    between rest and ctrl_high is still useful -- see Actuation, it is where
    antagonist release lives -- but it is not where neutral is.
    """
    data = mujoco.MjData(model)
    data.ctrl[:] = model.actuator_ctrlrange[:, 1]
    for _ in range(2000):
        mujoco.mj_step(model, data)
    return data.ten_length.copy()


class Channel:
    """One scalar DOF beyond the bend cursor.

    `pos` cables are pulled for s > 0, `neg` for s < 0. A counter-wound helical
    pair therefore becomes a single signed twist axis: push it one way to wind,
    the other to unwind.
    """

    def __init__(self, name: str, pos: np.ndarray, neg: np.ndarray | None = None):
        self.name = name
        self.pos = np.asarray(pos, int)
        self.neg = np.asarray(neg if neg is not None else [], int)


class Actuation:
    """Maps (cursor, axial, channels) -> tendon lengths for any variant.

    The bend ring generalises the POC's `convert_2d_cursor_to_target_lengths`:
    a cable pulls in proportion to how well its angular position aligns with
    the commanded direction. Everything else about the routing decides which
    cables belong to that ring and which become their own channel:

      * straight, full-length cables  -> the bend ring, driven by the cursor.
      * helical cables                -> a signed TWIST channel. They must not
        join the bend ring: their whole point is a moment arm about the
        backbone axis, so folding them into the cursor mapping would hide the
        one thing we are trying to look at.
      * cables terminating early      -> a SECTION channel, because bending only
        the proximal part independently of the tip is exactly the S-curve DOF a
        partial-length cable buys.

    The pull/release asymmetry is load-bearing and is kept from the POC: a
    negative alignment does not just mean "don't pull", it means "pay this
    cable OUT". Off-side cables lengthen as the tentacle bends, so a cable left
    at rest fights the motion; releasing it is what makes the achievable
    direction vary smoothly with the commanded one instead of snapping to the
    three cable axes.
    """

    def __init__(self, meta: dict, rest: np.ndarray):
        self.tendons = meta["tendons"]
        self.low = np.array(meta["ctrl_low"], float)
        self.high = np.array(meta["ctrl_high"], float)   # release headroom
        self.neutral = np.asarray(rest, float)           # zero-force length
        self.angles = np.radians([t["angle_deg"] for t in self.tendons])
        helix = np.array([t["helix_deg"] for t in self.tendons], float)
        t_end = np.round([t["t_end"] for t in self.tendons], 4)

        straight = np.flatnonzero(helix == 0.0)
        # bend ring = straight cables at the greatest reach; if a variant is
        # helical throughout (helix3), those cables carry the bend instead.
        pool = straight if straight.size else np.arange(len(helix))
        self.bend = pool[np.isclose(t_end[pool], t_end[pool].max())]

        self.channels: list[Channel] = []
        for te in sorted(set(t_end), reverse=True):
            at_te = np.flatnonzero(t_end == te)
            hel = at_te[helix[at_te] != 0.0]
            if hel.size:
                self.channels.append(Channel(
                    f"twist@{te:.2f}", hel[helix[hel] > 0], hel[helix[hel] < 0]))
            sec = np.setdiff1d(at_te[helix[at_te] == 0.0], self.bend)
            if sec.size:
                self.channels.append(Channel(f"section@{te:.2f}", sec))

    def __call__(self, cx: float, cy: float, a: float,
                 chan: np.ndarray) -> np.ndarray:
        """chan[i] in [-1, 1] drives self.channels[i].

        Demands ADD (then clip) because a cable can belong to more than one
        channel -- in a fully helical variant the same three cables carry both
        the bend and the twist, and letting the last writer win would silently
        cancel the cursor.
        """
        effect = np.zeros(len(self.neutral))
        mag = min(float(np.hypot(cx, cy)), 1.0)
        theta = np.arctan2(cy, cx)
        if mag > 1e-3:
            effect[self.bend] = np.cos(self.angles[self.bend] - theta) * mag

        for i, ch in enumerate(self.channels):
            s = float(np.clip(chan[i], -1.0, 1.0))
            idx = ch.pos if s >= 0 else ch.neg
            if idx.size == 0 or abs(s) < 1e-3:
                continue
            eff = (np.cos(self.angles[idx] - theta) * abs(s) if idx.size >= 3
                   else np.full(idx.size, abs(s)))   # a lone cable only pulls
            effect[idx] += eff

        effect = np.clip(effect, -1.0, 1.0)
        span = np.where(effect > 0, self.neutral - self.low, self.high - self.neutral)
        return np.clip(self.neutral - effect * span + a, self.low, self.high)


# =============================================================================
# pointer input
# =============================================================================
class Pointer:
    """Absolute cursor position -> (c_x, c_y) in the unit disk.

    Polled, not event-driven: a pynput Listener spawns a thread with its own
    macOS run loop, which is a needless thing to have fighting the GLFW loop
    under mjpython. Position can just be read. Scroll cannot, so scroll gets an
    optional listener that is allowed to fail (keys cover the same channel).
    """

    def __init__(self):
        from pynput import mouse
        self.ctl = mouse.Controller()
        self.scroll_accum = 0.0
        self.w, self.h = self._screen_size()
        self.half = min(self.w, self.h) / 2.0      # square region, centred
        self.listener = None
        try:
            self.listener = mouse.Listener(on_scroll=self._on_scroll)
            self.listener.start()
        except Exception as e:                     # noqa: BLE001
            print(f"  (scroll unavailable: {e}; use w/s for axial)")

    @staticmethod
    def _screen_size() -> tuple[float, float]:
        try:
            from Quartz import CGDisplayBounds, CGMainDisplayID
            b = CGDisplayBounds(CGMainDisplayID())
            return float(b.size.width), float(b.size.height)
        except Exception:                          # noqa: BLE001
            return 1440.0, 900.0

    def _on_scroll(self, x, y, dx, dy) -> None:
        self.scroll_accum += dy

    def read(self) -> tuple[float, float]:
        x, y = self.ctl.position
        cx = (x - self.w / 2.0) / self.half
        cy = -(y - self.h / 2.0) / self.half       # screen y is down
        mag = np.hypot(cx, cy)
        if mag > 1.0:                              # clamp into the unit disk
            cx, cy = cx / mag, cy / mag
        return float(cx), float(cy)

    def take_scroll(self) -> float:
        v, self.scroll_accum = self.scroll_accum, 0.0
        return v

    def stop(self) -> None:
        if self.listener:
            self.listener.stop()


# =============================================================================
# in-scene HUD (cv2/tk windows do not survive alongside GLFW under mjpython,
# so everything is drawn as line geometry inside the MuJoCo scene)
# =============================================================================
PALETTE = [(0.20, 0.45, 1.00, 1), (1.00, 0.55, 0.15, 1), (0.25, 0.80, 0.35, 1),
           (0.90, 0.30, 0.80, 1), (0.30, 0.85, 0.85, 1), (0.95, 0.85, 0.20, 1)]


class Hud:
    """Cable bars + tip trace + commanded-cursor dial, drawn in world space."""

    BAR_X0, BAR_DX = 0.13, 0.045      # bar field origin / spacing (m)
    BAR_Z0, BAR_H = -0.02, 0.20       # bar bottom / height (m)
    DIAL_R, DIAL_Z = 0.06, -0.06      # cursor dial radius / height (m)

    def __init__(self, n_tendons: int, low: np.ndarray, high: np.ndarray,
                 neutral: np.ndarray, labels: list[str] | None = None):
        self.n, self.low, self.high, self.neutral = n_tendons, low, high, neutral
        self.labels = labels or [str(k + 1) for k in range(n_tendons)]
        self.trace: deque[np.ndarray] = deque(maxlen=TRACE_MAX)
        self.show_trace = True
        self.ghost: list[np.ndarray] = []   # recorded tip path, for replay compare
        self.progress: float | None = None  # replay position in [0, 1], else None
        self._g = 0
        self._scn = None

    def set_ghost(self, tips) -> None:
        """Decimate a recorded tip path down to something drawable.

        Every segment costs a geom out of the same fixed budget as the live
        trace and the bars, and a take is thousands of frames long -- drawn
        whole it would silently starve everything queued after it.
        """
        pts = np.asarray(tips, float)
        if len(pts) > GHOST_MAX:
            pts = pts[np.linspace(0, len(pts) - 1, GHOST_MAX).astype(int)]
        self.ghost = [np.asarray(p, float) for p in pts]

    def _line(self, p0, p1, rgba, w=2.0) -> None:
        scn = self._scn
        if self._g >= scn.maxgeom:
            return
        gm = scn.geoms[self._g]
        mujoco.mjv_initGeom(gm, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3),
                            np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(gm, mujoco.mjtGeom.mjGEOM_LINE, w,
                             np.asarray(p0, np.float64), np.asarray(p1, np.float64))
        self._g += 1

    def _text(self, pos, text: str, rgba=(1, 1, 1, 1)) -> None:
        """Billboard text. mjGEOM_LABEL is the only decor geom that renders a
        string, so labels have to be their own geom rather than an attribute of
        the bar they annotate."""
        scn = self._scn
        if self._g >= scn.maxgeom:
            return
        gm = scn.geoms[self._g]
        mujoco.mjv_initGeom(gm, mujoco.mjtGeom.mjGEOM_LABEL,
                            np.array([0.02, 0.02, 0.02]), np.asarray(pos, np.float64),
                            np.eye(3).flatten(), np.asarray(rgba, np.float32))
        gm.label = text[:99]
        self._g += 1

    def _frac(self, k: int, length: float) -> float:
        return float((np.clip(length, self.low[k], self.high[k]) - self.low[k])
                     / max(self.high[k] - self.low[k], 1e-9))

    def draw(self, viewer, ctrl, ten_len, tip, cx, cy) -> None:
        if self.show_trace:
            self.trace.append(np.array(tip, float))
        with viewer.lock():
            self._scn = viewer.user_scn
            self._g = 0

            # --- cable bars: commanded (thick) vs actual length (tick) -------
            for k in range(self.n):
                x = self.BAR_X0 + k * self.BAR_DX
                bot = np.array([x, 0.0, self.BAR_Z0])
                top = np.array([x, 0.0, self.BAR_Z0 + self.BAR_H])
                self._line(bot, top, (0.45, 0.45, 0.45, 0.8), 1.2)
                self._line(bot + [-0.006, 0, 0], bot + [0.006, 0, 0],
                           (1.0, 0.25, 0.25, 0.9), 2.5)          # floor marker
                zn = self.BAR_Z0 + self._frac(k, self.neutral[k]) * self.BAR_H
                self._line([x - 0.006, 0, zn], [x + 0.006, 0, zn],
                           (0.9, 0.9, 0.3, 0.8), 2.0)   # rest: above = released

                z = self.BAR_Z0 + self._frac(k, ctrl[k]) * self.BAR_H
                slack = ten_len[k] < ctrl[k] - 1e-5              # cable not taut
                rgba = (0.55, 0.55, 0.55, 0.9) if slack else PALETTE[k % len(PALETTE)]
                self._line(bot, [x, 0.0, z], rgba, 6.0)          # commanded fill
                za = self.BAR_Z0 + self._frac(k, ten_len[k]) * self.BAR_H
                self._line([x - 0.007, 0, za], [x + 0.007, 0, za],
                           (1, 1, 1, 0.9), 2.0)                  # actual length
                # stagger every other label: mjGEOM_LABEL text is screen-space,
                # so neighbouring bars collide once you zoom out
                dz = -0.018 - 0.016 * (k % 2)
                self._text([x, 0.0, self.BAR_Z0 + dz], self.labels[k],
                           (0.55, 0.55, 0.55, 1) if slack
                           else PALETTE[k % len(PALETTE)])

            # annotate the tick meanings once, off the left end of the field
            xl = self.BAR_X0 - 0.060
            self._text([xl, 0.0, self.BAR_Z0 + self._frac(0, self.neutral[0])
                        * self.BAR_H], "rest", (0.9, 0.9, 0.3, 1))
            self._text([xl, 0.0, self.BAR_Z0], "floor", (1.0, 0.35, 0.35, 1))
            self._text([xl, 0.0, self.BAR_Z0 + self.BAR_H], "released",
                       (0.6, 0.6, 0.6, 1))

            # --- replay progress, under the bar field --------------------------
            if self.progress is not None:
                x0 = self.BAR_X0 - 0.010
                x1 = self.BAR_X0 + (self.n - 1) * self.BAR_DX + 0.010
                zp = self.BAR_Z0 - 0.055
                self._line([x0, 0, zp], [x1, 0, zp], (0.4, 0.4, 0.4, 0.7), 2.0)
                self._line([x0, 0, zp],
                           [x0 + (x1 - x0) * self.progress, 0, zp],
                           (1.0, 0.9, 0.2, 1.0), 5.0)

            # --- commanded cursor dial ---------------------------------------
            c = np.array([0.0, 0.0, self.DIAL_Z])
            ring = [c + [self.DIAL_R * np.cos(t), self.DIAL_R * np.sin(t), 0]
                    for t in np.linspace(0, 2 * np.pi, 33)]
            for p, q in zip(ring, ring[1:]):
                self._line(p, q, (0.5, 0.5, 0.5, 0.6), 1.2)
            self._line(c, c + [cx * self.DIAL_R, cy * self.DIAL_R, 0],
                       (1.0, 0.9, 0.2, 1.0), 4.0)

            # --- ghost (recorded) path, then the live trace on top -------------
            if len(self.ghost) > 1:
                for p, q in zip(self.ghost, self.ghost[1:]):
                    self._line(p, q, (0.95, 0.55, 0.15, 0.5), 1.5)

            # --- tip trace ----------------------------------------------------
            if self.show_trace and len(self.trace) > 1:
                pts = list(self.trace)
                for i in range(1, len(pts)):
                    f = i / len(pts)                             # fade with age
                    self._line(pts[i - 1], pts[i], (0.2, 1.0, 0.6, 0.15 + 0.85 * f), 2.0)

            self._scn.ngeom = self._g


# =============================================================================
def total_twist_deg(model, data) -> float:
    """Accumulated MATERIAL twist along the backbone, in degrees.

    Not the tip frame's apparent rotation -- that conflates twist with bend
    (tilt the tip far enough and any fixed world reference projects into
    nonsense, which is how a straight-cable model can appear to 'roll' 180 deg
    while carrying no torsion at all).

    Instead: swing-twist decomposition. Each ball joint's quaternion relative to
    its parent is split into a swing (bend) and a rotation about its own local
    z; the twist part is 2*atan2(qz, qw). Summing along the chain gives the
    torsion actually stored in the body, which is invariant to how bent it is
    and is exactly what a helical cable can drive and a parallel one cannot.
    """
    total = 0.0
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_BALL:
            continue
        w, _, _, z = data.qpos[model.jnt_qposadr[j]: model.jnt_qposadr[j] + 4]
        total += 2.0 * np.arctan2(z, w)
    return float(np.degrees(total))


def run_probe(model, data, act: Actuation, mag: float = 0.8) -> None:
    """Sweep the cursor around a circle; report commanded vs achieved azimuth.

    Worth running once per variant: the POC's control angles (90/-30/210 deg)
    are both offset and opposite in handedness from the simulated cable
    positions (60/180/300 deg), so 'cursor +y' is NOT 'tip bends +y'. The
    policy learned around that, but a human authoring gestures needs to know.
    """
    print("\n  probe: commanded azimuth -> achieved tip azimuth "
          f"(|cursor| = {mag})")
    print(f"  {'cmd deg':>8} {'tip deg':>8} {'delta':>7} {'reach m':>8} {'roll deg':>9}")
    saved = data.ctrl.copy()
    sec = np.zeros(max(len(act.channels), 1))
    rows = []
    for cmd in range(0, 360, 30):
        th = np.radians(cmd)
        data.ctrl[:] = act(mag * np.cos(th), mag * np.sin(th), 0.0, sec)
        for _ in range(1500):
            mujoco.mj_step(model, data)
        tip = data.site_xpos[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "tip_center")]
        got = np.degrees(np.arctan2(tip[1], tip[0])) % 360
        delta = (got - cmd + 180) % 360 - 180
        rows.append(delta)
        print(f"  {cmd:8d} {got:8.1f} {delta:+7.1f} {np.hypot(*tip[:2]):8.4f} "
              f"{total_twist_deg(model, data):9.2f}")
    print(f"  mean offset {np.mean(rows):+.1f} deg, spread {np.ptp(rows):.1f} deg "
          f"-- a constant offset is a frame rotation; a large spread means the "
          f"mapping is direction-dependent.\n")
    data.ctrl[:] = saved
    mujoco.mj_resetData(model, data)


# =============================================================================
# replay
# =============================================================================
def resolve_gesture(spec: str | None, which: str) -> Path | None:
    """Find a take. `None`/'last' means the newest, preferring this variant."""
    if spec in (None, "last", "latest"):
        pool = sorted(GESTURES.glob(f"{which}_*.npz"),
                      key=lambda p: p.stat().st_mtime)
        if not pool:
            pool = sorted(GESTURES.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        return pool[-1] if pool else None
    for cand in (Path(spec), GESTURES / spec, GESTURES / f"{spec}.npz"):
        if cand.exists():
            return cand
    return None


def load_gesture(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        g = {k: z[k] for k in ("t", "cursor", "axial", "ctrl", "tip")}
    g["name"] = path.name
    return g


class Player:
    """Replays a recorded take by feeding its `ctrl` rows back in.

    Replays the recorded CABLE COMMANDS, not the cursor. The cursor is only one
    of the inputs that produced them -- the extra-channel state is not stored in
    the take, so re-deriving through Actuation would quietly drop any twist or
    section channel the take was driven with. Cursor is carried along anyway so
    the dial animates.

    Sampled by interpolation on the recorded timestamps rather than by frame
    index, so a take recorded on a loop that missed its deadline still replays
    at the speed it was performed at.
    """

    def __init__(self, g: dict, loop: bool = False):
        self.t = np.asarray(g["t"], float) - float(g["t"][0])
        self.ctrl = np.asarray(g["ctrl"], float)
        self.cursor = np.asarray(g["cursor"], float)
        self.tip = np.asarray(g["tip"], float)
        self.name = g.get("name", "?")
        self.loop = loop
        self.clock = 0.0
        self.done = False

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def sample(self, dt: float) -> tuple[np.ndarray, float, float, float]:
        u = min(self.clock, self.duration)
        ctrl = np.array([np.interp(u, self.t, self.ctrl[:, k])
                         for k in range(self.ctrl.shape[1])])
        cx = float(np.interp(u, self.t, self.cursor[:, 0]))
        cy = float(np.interp(u, self.t, self.cursor[:, 1]))
        frac = u / max(self.duration, 1e-9)
        # end on the frame that lands ON the last timestamp, not the one after
        # it: duration is rarely an exact multiple of dt, so testing the
        # advanced clock instead would drop the final pose of every take.
        at_end = u >= self.duration
        self.clock += dt
        if at_end:
            if self.loop:
                self.clock = 0.0
            else:
                self.done = True
        return ctrl, cx, cy, float(min(frac, 1.0))


def parse_args(argv: list[str]) -> tuple[str, str | None, bool, bool]:
    """-> (variant, play_spec, autoplay, loop). Bare positional stays the variant."""
    which, play, autoplay, loop = "baseline", None, False, False
    pos: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--play", "-p"):
            autoplay = True
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            if nxt and not nxt.startswith("-"):
                play, i = nxt, i + 1
            else:
                play = "last"
        elif a == "--loop":
            loop = True
        elif a in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        elif a.startswith("-"):
            sys.exit(f"unknown option {a!r} (try --help)")
        else:
            pos.append(a)
        i += 1
    if pos:
        which = pos[0]
    return which, play, autoplay, loop


# =============================================================================
def main() -> None:
    which, play_spec, autoplay, loop = parse_args(sys.argv[1:])
    model, meta = load_model(which)
    data = mujoco.MjData(model)
    act = Actuation(meta, measure_rest(model))
    dt = model.opt.timestep

    print(__doc__)
    print(f"  model: {which}   cables: {model.nu}   dt: {dt}")
    for k, t in enumerate(act.tendons):
        role = "bend" if k in act.bend else next(
            (c.name for c in act.channels if k in c.pos or k in c.neg), "-")
        print(f"    cable {k+1}: angle {t['angle_deg']:+7.1f} deg  "
              f"helix {t['helix_deg']:+6.1f} deg  ends at {t['t_end']*100:5.1f}%  "
              f"rest {act.neutral[k]:.4f}  role {role}")
    if act.channels:
        print("  extra channels (select with 1..9, adjust with , / .): "
              + ", ".join(f"{i+1}={c.name}" for i, c in enumerate(act.channels)))
    print()

    ptr = Pointer()
    bar_labels = []
    for k in range(model.nu):
        role = "bend" if k in act.bend else next(
            (c.name.split("@")[0] for c in act.channels if k in c.pos or k in c.neg),
            "-")
        bar_labels.append(f"{k + 1} {role}")
    hud = Hud(model.nu, act.low, act.high, act.neutral, bar_labels)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_center")

    st = {"axial": 0.0, "frozen": False, "frozen_c": (0.0, 0.0),
          "mode": "CURSOR", "sel": 0,
          "chan": np.zeros(max(len(act.channels), 1)),
          "raw": act.neutral.copy(), "rec": None, "probe": False, "quit": False,
          "gesture": None, "play": None, "play_req": False, "loop": loop}

    if play_spec is not None or autoplay:
        path = resolve_gesture(play_spec, which)
        if path is None:
            print(f"  (no take to replay in {GESTURES}; record one with r)")
        else:
            st["gesture"] = load_gesture(path)
            st["play_req"] = True            # starts on the first loop pass
            n = len(st["gesture"]["t"])
            dur = float(st["gesture"]["t"][-1] - st["gesture"]["t"][0])
            print(f"  loaded {path.name}: {n} frames, {dur:.1f}s "
                  f"-- ENTER replays it again\n")

    def begin_playback() -> None:
        g = st["gesture"]
        if g is None:
            print("\n  -> nothing loaded. Record with r, or start with --play")
            return
        if g["ctrl"].shape[1] != model.nu:
            print(f"\n  -> take has {g['ctrl'].shape[1]} cables, model '{which}' "
                  f"has {model.nu}. Replay it on the variant it was recorded on.")
            return
        # Settle into the take's OWN first command, not the neutral pose. A
        # recording almost never starts from rest -- you drive somewhere, then
        # hit r -- so resetting to neutral prepends a transient that is not in
        # the take: measured 98 mm of tip error at frame 0, decaying over ~0.4 s.
        # Settling on ctrl[0] reproduces the take's starting equilibrium and
        # takes that to 0.004 mm median. It is also what a gesture player has to
        # do on hardware anyway: reach the start pose before running the motion.
        mujoco.mj_resetData(model, data)
        data.ctrl[:] = g["ctrl"][0]
        for _ in range(400):
            mujoco.mj_step(model, data)
        hud.trace.clear()
        hud.set_ghost(g["tip"])
        st["play"] = Player(g, loop=st["loop"])
        st["frozen"] = False
        print(f"\n  -> replaying {g['name']} ({st['play'].duration:.1f}s)"
              f"{' [loop]' if st['loop'] else ''}")

    def key_callback(keycode: int) -> None:
        if keycode in (KEY_ENTER, KEY_KP_ENTER):
            if st["play"] is not None:
                st["play"] = None            # pure state, safe from any thread
                hud.progress = None
                print("\n  -> replay cancelled")
            else:
                # DEFERRED, like the probe below: key_callback runs on the
                # viewer's thread, and begin_playback resets and steps mjData.
                # Doing that here races the viewer copying mjData to render and
                # aborts the process with "stack is in use".
                st["play_req"] = True
            return
        # arrows first: they are GLFW codes outside the printable range, so
        # they must not fall through to the chr() branches below
        if keycode == KEY_UP:
            st["axial"] = min(AXIAL_LIMIT, st["axial"] + AXIAL_STEP)
            return
        if keycode == KEY_DOWN:
            st["axial"] = max(-AXIAL_LIMIT, st["axial"] - AXIAL_STEP)
            return

        ch = chr(keycode).lower() if 0 <= keycode < 0x110000 else ""
        if keycode == ord(" "):
            st["frozen"] = not st["frozen"]
            print(f"  -> {'frozen' if st['frozen'] else 'live'}")
        elif ch == "0":
            st["axial"] = 0.0
        elif ch == "m":
            st["mode"] = "RAW" if st["mode"] == "CURSOR" else "CURSOR"
            st["raw"] = act.neutral.copy()
            print(f"  -> mode {st['mode']}")
        elif ch.isdigit() and ch != "0":
            i = int(ch) - 1
            n_sel = model.nu if st["mode"] == "RAW" else len(act.channels)
            if i >= n_sel:
                print(f"\n  -> nothing at {i + 1} ({st['mode']} mode has {n_sel})")
            else:
                st["sel"] = i
                name = (f"cable {i + 1}" if st["mode"] == "RAW"
                        else act.channels[i].name)
                print(f"\n  -> selected {name}")
        elif ch in ",.":
            sign = 1 if ch == "." else -1
            if st["mode"] == "RAW" and st["sel"] < model.nu:
                k = st["sel"]
                st["raw"][k] = np.clip(st["raw"][k] + sign * AXIAL_STEP,
                                       act.low[k], act.high[k])
                print(f"\n  -> cable {k+1} = {st['raw'][k]:.4f}")
            elif st["sel"] < len(act.channels):
                st["chan"][st["sel"]] = np.clip(
                    st["chan"][st["sel"]] + sign * CHAN_STEP, -1.0, 1.0)
                print(f"\n  -> {act.channels[st['sel']].name} = "
                      f"{st['chan'][st['sel']]:+.2f}")
        elif ch == "/":
            if st["mode"] == "CURSOR" and st["sel"] < len(act.channels):
                st["chan"][st["sel"]] = 0.0
                print(f"\n  -> {act.channels[st['sel']].name} = 0 (reset)")
            elif st["mode"] == "RAW":
                st["raw"] = act.neutral.copy()
                print("\n  -> all cables back to rest")
        elif ch == "t":
            hud.show_trace = not hud.show_trace
        elif ch == "c":
            hud.trace.clear()
            hud.ghost = []
        elif ch == "l":
            st["loop"] = not st["loop"]
            if st["play"] is not None:
                st["play"].loop = st["loop"]
            print(f"\n  -> replay loop {'on' if st['loop'] else 'off'}")
        elif ch == "p":
            st["probe"] = True
        elif ch == "r":
            if st["rec"] is None:
                st["rec"] = []
                print("  -> RECORDING")
            else:
                path = save_recording(st["rec"], which)
                st["rec"] = None
                if path is not None:      # becomes the take ENTER replays
                    st["gesture"] = load_gesture(path)
                    print("  -> ENTER to replay it")
        elif ch == "q":
            st["quit"] = True

    data.ctrl[:] = act.neutral
    for _ in range(200):
        mujoco.mj_step(model, data)

    last_print = 0.0
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running() and not st["quit"]:
            tic = time.perf_counter()

            if st["probe"]:
                run_probe(model, data, act)
                st["probe"] = False

            if st["play_req"]:               # see key_callback: must run here
                st["play_req"] = False
                begin_playback()

            st["axial"] = float(np.clip(
                st["axial"] + ptr.take_scroll() * AXIAL_STEP,
                -AXIAL_LIMIT, AXIAL_LIMIT))

            if st["play"] is not None:
                # replay owns the cables; live input is ignored until it ends
                ctrl, cx, cy, hud.progress = st["play"].sample(dt)
                if st["play"].done:
                    print(f"\n  -> replay done ({st['play'].name})")
                    st["play"] = None
                    hud.progress = None
            elif st["mode"] == "RAW":
                cx = cy = 0.0
                ctrl = np.clip(st["raw"] + st["axial"], act.low, act.high)
            else:
                if st["frozen"]:
                    cx, cy = st["frozen_c"]
                else:
                    cx, cy = ptr.read()
                    st["frozen_c"] = (cx, cy)
                ctrl = act(cx, cy, st["axial"], st["chan"])

            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            tip = data.site_xpos[tip_id].copy()

            try:
                hud.draw(viewer, ctrl, data.ten_length, tip, cx, cy)
            except Exception as e:                 # noqa: BLE001
                print(f"  (hud off: {e})")
                hud.show_trace = False

            if st["rec"] is not None and st["play"] is None:
                st["rec"].append((data.time, cx, cy, st["axial"],
                                  *ctrl, *tip, total_twist_deg(model, data)))

            now = time.perf_counter()
            if now - last_print > 0.2:
                last_print = now
                slack = "".join("." if data.ten_length[k] < ctrl[k] - 1e-5 else "#"
                                for k in range(model.nu))
                label = "PLAY" if st["play"] is not None else st["mode"]
                print(f"\r  {label:6} c=({cx:+.2f},{cy:+.2f}) a={st['axial']:+.3f} "
                      f"taut[{slack}] tip=({tip[0]:+.3f},{tip[1]:+.3f},{tip[2]:.3f}) "
                      f"roll={total_twist_deg(model, data):+6.1f}deg"
                      f"{'  REC' if st['rec'] is not None else '    '}",
                      end="", flush=True)

            viewer.sync()
            left = dt - (time.perf_counter() - tic)
            if left > 0:
                time.sleep(left)

    ptr.stop()
    if st["rec"]:
        save_recording(st["rec"], which)
    print("\n  done.")


def save_recording(rows: list, which: str) -> Path | None:
    if not rows:
        print("  -> nothing recorded")
        return None
    arr = np.asarray(rows, float)
    n_ctrl = arr.shape[1] - 8
    path = GESTURES / f"{which}_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez(path, t=arr[:, 0], cursor=arr[:, 1:3], axial=arr[:, 3],
             ctrl=arr[:, 4:4 + n_ctrl], tip=arr[:, 4 + n_ctrl:7 + n_ctrl],
             roll=arr[:, -1], variant=which)
    print(f"\n  -> saved {len(rows)} frames ({arr[-1, 0]:.1f}s) to {path}")
    return path


if __name__ == "__main__":
    main()
