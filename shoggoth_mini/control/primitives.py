"""Motion primitives and behaviors (ported from legacy action_normalized.py)."""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional
import numpy as np
import typer
from rich.console import Console

from .geometry import cursor_to_motor_positions
from ..hardware.motors import MotorController
from ..common.constants import (
    MOTOR_NORMALIZED_POSITIONS,
    MOTOR_ONE_FULL_TURN_TICKS,
)
from ..configs import get_hardware_config

console = Console()
app = typer.Typer(help="Motion primitive utilities")
logger = logging.getLogger(__name__)


class MotionBehavior(Enum):
    """Enumeration of available motion behaviors."""

    YES = "<yes>"
    NO = "<no>"
    SHAKE = "<shake>"
    CIRCLE = "<circle>"
    SLOW_CIRCLE = "<slow_circle>"
    PACKET_AM = "<wave_packet_am>"
    PACKET_FM = "<wave_packet_fm>"
    SLOW_BREATHE = "<slow_breathe>"
    NORMAL_BREATHE = "<normal_breathe>"
    ARCHED = "<arched>"
    UNARCH = "<unarch>"
    SIDE_SIDE = "<side_side>"
    SIDE_SIDE_FAST = "<side_side_fast>"
    GRAB = "<grab_object>"
    RELEASE = "<release_object>"
    HIGH_FIVE = "<high_five>"

    @classmethod
    def from_action_string(cls, action_string: str) -> Optional["MotionBehavior"]:
        """Get behavior from action string.

        Args:
            action_string: The action string (e.g., "<yes>", "<grab_object>")

        Returns:
            MotionBehavior enum value if found, None otherwise
        """
        try:
            return cls(action_string)
        except ValueError:
            return None


@dataclass
class YesMotionConfig:
    """Configuration for yes/nodding motion - handcrafted for natural movement."""

    down_position: np.ndarray = field(default_factory=lambda: np.array([0.12, -0.08]))
    center_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    hold_duration: float = 0.13


@dataclass
class NoMotionConfig:
    """Configuration for no/head-shake motion - handcrafted for natural movement."""

    left_position: np.ndarray = field(default_factory=lambda: np.array([0.15, 0.0]))
    down_position: np.ndarray = field(default_factory=lambda: np.array([0.12, -0.08]))
    right_position: np.ndarray = field(default_factory=lambda: np.array([-0.0, -0.15]))
    # The lead-in move, not a pause: it is the sleep AFTER the first
    # set_positions, so it is the time that step has to complete in.
    #
    # It was 0.05, which asked 839 ticks in 50 ms = ~16,800 ticks/s against a
    # measured servo ceiling of ~7,600 (char_primitive_sweep, 2026-09-04). At
    # more than twice the ceiling the move is still in flight when the next
    # command lands, so the shape came from servo dynamics rather than from the
    # waypoints -- the same defect slow_circle was reworked for.
    #
    # 0.13 matches hold_duration and gives ~6,450 ticks/s. The minimum that
    # clears the ceiling is 839/7600 = 0.11 s, so this leaves margin. The cost
    # is 80 ms of extra lead-in, which is nothing beside the seconds of
    # perception and gesture latency ahead of it.
    initial_delay: float = 0.13
    hold_duration: float = 0.13


@dataclass
class ShakeMotionConfig:
    """Configuration for shake motion - handcrafted for natural movement."""

    left_position: np.ndarray = field(default_factory=lambda: np.array([0.04, 0.07]))
    right_position: np.ndarray = field(default_factory=lambda: np.array([-0.04, -0.07]))
    hold_duration: float = 0.17


@dataclass
class CircleMotionConfig:
    """Configuration for circular motion - handcrafted for smooth movement."""

    radius: float = 0.07
    points_per_circle: int = 20
    time_per_point: float = 0.009


@dataclass
class SlowCircleMotionConfig:
    """Circle sized to what the servos can actually deliver.

    The stock circle asks for more than the hardware can do, in two separate
    ways (measured 2026-08-26 with tools/char_primitive_sweep.py --dry-run):

        entry jump   248 ticks in 9 ms  = 27,556 ticks/s   3.6x the ceiling
        steady state  87 ticks in 9 ms  =  9,667 ticks/s   1.3x the ceiling

    against a measured servo ceiling of ~7600 ticks/s. The consequence is that
    the traced shape is set by servo dynamics rather than by the waypoints, so
    tuning radius or points does much less than it appears to, and the entry is
    a slam from neutral rather than a curve.

    This version keeps the same radius but triples the point count (smaller
    steps), dwells longer at each, and eases in and out of the circle instead
    of jumping. At these values the peak rate is about 1500 ticks/s — a fifth
    of the ceiling — so the servo tracks the path instead of chasing it.
    """

    radius: float = 0.07
    points_per_circle: int = 60
    time_per_point: float = 0.02
    revolutions: int = 2
    # Points spent spiralling out to the radius, and back in again. The ease is
    # a SPIRAL rather than a radial line: the angle keeps advancing while the
    # radius changes, so the path never changes direction abruptly. An earlier
    # version held angle 0 and moved straight in and out, which put a corner in
    # the path exactly where the motion was supposed to be smoothest.
    entry_steps: int = 15
    # Below cursor_to_motor_positions' 0.01 deadzone every radius resolves to
    # the calibrated position, so starting the spiral at 0 wastes commands that
    # do nothing. Start just above it.
    min_radius: float = 0.012


@dataclass
class WavePacketAMConfig:
    """Gaussian wave packet, amplitude-modulated — the textbook form.

        cursor(t) = amplitude * exp(-0.5 (t/sigma)^2) * sin(2 pi f t) * direction

    A sway along one axis that grows out of nothing, peaks, and fades. It needs
    no explicit ease: the envelope starts and ends near zero by construction.

    Sizing is set by the rate budget. Peak commanded rate is
    amplitude * 4096 * 2 pi f ticks/s, and the servos top out near 7600. At
    0.18 and 0.6 Hz that is ~2780 ticks/s, roughly a third of the ceiling, so
    the servo tracks the waveform instead of chasing it. Raising either term
    requires lowering the other.
    """

    amplitude: float = 0.18
    frequency_hz: float = 0.6
    sigma_s: float = 1.5
    # Total duration in units of sigma, centred on the peak. 6 sigma captures
    # 99.7% of the envelope, so the ends really are at zero.
    span_sigmas: float = 6.0
    direction_deg: float = 330.0     # motor 2's axis: toward the camera
    time_per_point: float = 0.02

    # LINEAR (False): x and y are driven in phase, so the cursor oscillates along
    # direction_deg and the path is a straight line. On a pure tendon axis this
    # gives the 1 : -0.5 : -0.5 split, i.e. two motors move identically.
    #
    # CIRCULAR (True): x and y are put in QUADRATURE — 90 degrees apart — so the
    # cursor rotates while the Gaussian scales its radius, spiralling out and
    # back in. In motor space that is the three tendons driven at 120-degree
    # phase offsets, which is the same statement: the axes are already 120 apart,
    # so a rotating cursor projects onto them as three phase-shifted sinusoids.
    circular: bool = True

    # Number of packets played back to back. The phase ACCUMULATES across
    # repeats rather than resetting, so the rotation continues in the same
    # direction through each seam; the envelope is what restarts. Because the
    # envelope is ~0 at both ends of a packet, position is continuous there too.
    repeats: int = 5


@dataclass
class WavePacketFMConfig:
    """Gaussian wave packet, frequency-modulated — a circle that chirps.

        f(t)     = f_min + (f_max - f_min) * exp(-0.5 (t/sigma)^2)
        theta(t) = 2 pi * integral of f
        cursor   = radius * (cos theta, sin theta)

    Constant radius, varying angular rate: slow, accelerating to a peak, then
    slowing again. The phase is INTEGRATED step by step rather than evaluated
    as 2 pi f(t) t — the latter is a different signal and is discontinuous
    wherever f changes.

    Because the radius is constant, this one does need an ease, otherwise it
    starts with a jump from neutral onto the circle. It spirals in and out, as
    slow_circle does.
    """

    # Sped up 1.35x on 2026-08-28. A uniform time-scaling t -> 1.35t: sigma
    # shrinks by that factor and both frequencies rise by it, so the shape is
    # identical and only the playback rate changes. Scaling sigma alone would
    # shorten the packet without making it move faster; scaling frequency alone
    # would speed the rotation without shortening it.
    #   was  sigma 2.0, f 0.15-0.70 Hz, 12.2 s, peak 2611 ticks/s
    #   now  sigma 1.48, f 0.20-0.95 Hz,  9.0 s, peak ~3550 ticks/s
    radius: float = 0.15
    f_min_hz: float = 0.2025
    f_max_hz: float = 0.945
    sigma_s: float = 1.4815
    span_sigmas: float = 6.0
    time_per_point: float = 0.02
    entry_steps: int = 15
    min_radius: float = 0.012      # above cursor_to_motor_positions' deadzone

    # Radius envelope.
    #   "triangle" : ramps linearly 0 -> radius at the midpoint -> 0, so the
    #                packet grows and shrinks as well as speeding up and slowing
    #                down. Both peak together at the centre.
    #   "constant" : full radius throughout, with a short spiral ease at each end
    #                (the original form).
    # The triangle takes the radius to ~0 at both ends, which is inside
    # cursor_to_motor_positions' 0.01 deadzone, so this primitive lowers that
    # deadzone — otherwise the first and last ~2 s would snap flat to the
    # calibrated position instead of tapering.
    envelope: str = "triangle"

    # Number of packets played back to back, phase accumulating across the seams
    # exactly as for the AM packet.
    repeats: int = 5


@dataclass
class SlowBreatheConfig:
    """An undulating grab: motor 2 swelling from neutral to a peak and back.

        offset_2(t) = peak_ticks * (1 - cos(2 pi t / period)) / 2

    UNIPOLAR — it never goes below neutral, so it reads as a slow squeeze and
    release rather than a sway through centre. Motors 1 and 3 pay out half as
    much each, which is what any motion along a single tendon axis does.

    A raised cosine rather than |sin|: its derivative is zero at both the top
    and the bottom, so the breath eases in and out instead of cornering at the
    neutral end.

    peak_ticks is meaningful as ticks only because direction_deg is exactly
    motor 2's axis, where that motor's alignment is 1.0 and the offset is
    magnitude * 4096. Off-axis it would scale by the cosine of the difference.

    Peak rate is peak_ticks * pi / period = 550 ticks/s at these values, about
    7% of the servo ceiling — the gentlest primitive in the set.
    """

    peak_ticks: int = 700
    period_s: float = 4.0
    repeats: int = 5
    direction_deg: float = 330.0     # motor 2's axis, toward the camera
    time_per_point: float = 0.02


@dataclass
class GrabMotionConfig:
    """Configuration for grab/release motions - handcrafted positions."""

    grab_cursor_pos: np.ndarray = field(
        # Upstream ships 0.7. This is 0.28 — see the sizing note below.
        #
        # HISTORY, because the stated reason for the first change was wrong:
        # it was reduced to 0.25 on 2026-08-14 with the explanation that these
        # servos take a modulo-4096 SHORTEST PATH, so a command over 2048 ticks
        # executes backwards. THAT EXPLANATION WAS WRONG. Test C swept +/-50 to
        # +/-2000 and test D/E ran 0 to 4150: no reversal anywhere, and no special
        # behaviour at 2048. The probe that "confirmed" it 11/11 was run against
        # a calibration whose motor-2 zero was -1548, so
        # every one of its targets was a negative absolute position and it was
        # measuring the sign-magnitude encoding, not a shortest-path rule. Treat
        # that probe's results as void.
        #
        # What is actually true of 0.7, and is reason enough for caution: it is a
        # single UNRAMPED command of +2867 ticks on motor 2 — 77 mm of cable in
        # one move at roughly 200 mm/s — while motors 1 and 3 each pay out 38.5 mm
        # simultaneously. From a 2048 zero it targets 4915, outside 0..4095. Paying
        # cable out fast with no tension on it is what strips wire off the rollers.
        #
        # SET TO 0.28 (2026-08-26). Sized for RETENSION HEADROOM, not for depth.
        #
        # This value tracks the calibrated zero and must be rechecked whenever
        # the zero moves. With zeros at 2722/2684/2944 the arithmetic maximum is
        # 0.336; 0.28 keeps ~200 ticks spare. Run
        #   python tools/char_primitive_sweep.py --dry-run
        # after any retension: it prints the largest magnitude that still fits.
        #
        # Previously 0.41, sized against zeros near 2168/2703/2166.
        #
        # Grab winds motor 2 in further than any other motion, so it is the first
        # thing to run out of range when the calibrated zero moves — and the zero
        # only ever moves UP, because retension is always wind-in to take up slack.
        # Measured with tools/char_primitive_sweep.py --calib-offset:
        #
        #   magnitude 0.48, zero +0   -> target 4014   ok, 81 ticks spare
        #   magnitude 0.48, zero +100 -> target 4114   REFUSED, over by 19
        #   magnitude 0.48, zero +300 -> target 4314   REFUSED, over by 219
        #   magnitude 0.41, zero +300 -> target 4027   ok
        #
        # 0.48 was the largest value fitting inside 4095 with the ~36-tick
        # overshoot from test C, and that is exactly why it was wrong: it spent
        # the entire budget on grab depth and left none for the zero to move.
        # 0.41 reserves ~300 ticks (8 mm of cable) of retension freedom, at a cost
        # of about 3.5 mm of grab depth. The two are directly exchangeable:
        # max magnitude = (4095 - 36 - zero - retension_budget) / 4096.
        #
        # For reference on the boundary itself:
        #   0.50 -> target 4095, overshoots to ~4131   CROSSES THE FOLD
        #   0.70 -> target 4915 (upstream value)       far past it
        #
        # Crossing the fold is the actual failure mode, measured 2026-08-18.
        # Past 4095 the reading comes back negative, and every read-modify-write
        # path in the stack (idle's position tracking, closed_loop's read-back,
        # any homing ramp) then computes a negative Goal_Position. A raw negative
        # is decoded SIGN-MAGNITUDE, so -3217 becomes a target of -29551 and the
        # motor runs away at full speed until it jams. That is what tore tendon 2
        # off its roller, not any shortest-path rule.
        #
        # Raising this further requires fold-safe position tracking everywhere,
        # not just a bigger number here.
        default_factory=lambda: MOTOR_NORMALIZED_POSITIONS["2"] * 0.28
    )
    hold_duration: float = 0.3


@dataclass
class ReleaseMotionConfig:
    """Configuration for release motion - handcrafted for natural movement."""

    neutral_cursor_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    hold_duration: float = 0.3


@dataclass
class HighFiveMotionConfig:
    """Configuration for high five motion - handcrafted for natural movement."""

    high_five_position: np.ndarray = field(
        default_factory=lambda: np.array([0.10392, -0.06])
    )
    center_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    hold_duration: float = 0.06


YES_CONFIG = YesMotionConfig()
NO_CONFIG = NoMotionConfig()
SHAKE_CONFIG = ShakeMotionConfig()
CIRCLE_CONFIG = CircleMotionConfig()
SLOW_CIRCLE_CONFIG = SlowCircleMotionConfig()
PACKET_AM_CONFIG = WavePacketAMConfig()
PACKET_FM_CONFIG = WavePacketFMConfig()
SLOW_BREATHE_CONFIG = SlowBreatheConfig()
# Same breath at double frequency, for the state machine's NOTICING state --
# present but not yet attended to, so a touch more alive than ALONE without
# being a reaction. Only the period changes; keeping it a config variant rather
# than a second primitive means the shape, the easing and the deadzone
# workaround stay defined in exactly one place.
#
# Peak rate scales with 1/period: 700 * pi / 2.0 = ~1100 ticks/s, about 14% of
# the measured 7600 ticks/s servo ceiling, so it is still a gentle motion.
NORMAL_BREATHE_CONFIG = SlowBreatheConfig(period_s=2.0)


@dataclass
class ArchedConfig:
    """A held arch: ramp to a fixed offset along motor 2's axis and stay there.

    Written for SAD and ANGRY, which previously used grab. Grab reaches as deep
    as the calibration allows -- 1127 ticks at the 2026-09-04 zeros, arriving in
    a single command at 3700 ticks/s. That is a lunge, and for a state whose
    whole point is a held posture it is both the wrong shape and the closest
    thing in the set to the encoder end stop.

    This is the opposite: a FIXED 700 ticks, reached by a linear ramp over 1.5 s.
    Fixed rather than derived because the pose is the point -- at 700 ticks it
    sits 528 ticks clear of the range at today's zeros and would still be clear
    after 500 ticks of retension drift, so it needs no cap to stay safe. Should
    the zeros ever climb that far, check_limits reports it before anything moves.

    Peak rate is 700/1.5 = 467 ticks/s, about 6% of the servo ceiling and among
    the gentlest motions in the set.

    Holds when it finishes, like grab: MotionWorker plays the release when the
    state changes.
    """

    offset_ticks: int = 700
    ramp_s: float = 1.5
    direction_deg: float = 330.0     # motor 2's axis, as slow_breathe uses
    time_per_point: float = 0.02


ARCHED_CONFIG = ArchedConfig()


@dataclass
class SideSideConfig:
    """A tick-tock sway: sweep to one side, then a fast retreat and overshoot.

    Anticipation and overshoot, which is what makes a movement read as intended
    rather than merely executed. The sweep is a raised cosine, so it arrives at
    each extreme with zero velocity; the flourish then punctuates that pause
    with a beat at `speed_mult` times the sweep's pace. The eye reads the CHANGE
    in speed more than the distance, which is why a 20% wobble is legible while
    covering a fifth of the sweep.

    LEFT-RIGHT, not fore-aft. `direction_deg` 60 is perpendicular to motor 2's
    axis, so motor 2 stays exactly still (alignment 0.000) and motors 1 and 3
    oppose each other at +/-0.866. Using motor 2's own axis, as the breaths do,
    would be a nod rather than a sway.

    STARTS AND ENDS AT NEUTRAL, which is what makes it loop. It leads in from
    centre over half a sweep and leads out the same way, so the last command of
    one play and the first of the next are both the calibrated pose, with zero
    velocity at each. execute_behavior's reset afterwards is then a no-op rather
    than a visible snap.

    Retreat and overshoot are separate: it pulls back 30% and then goes 20%
    PAST the extreme, so the two beats cover 0.30 and 0.50 of the amplitude.
    Their durations are proportional to those distances, which is what keeps
    both at the same speed -- a shared duration would have made the overshoot
    travel 1.7x further in the same time and read as a lunge rather than the
    second half of one gesture.

    Segments are a list of (target as a fraction of amplitude, duration) rather
    than a closed form, because every number here is one a person chose: 0.70
    the retreat, 1.20 the overshoot, durations following from speed_mult. A sum
    of sinusoids would bury all three in phase relationships.

    The beats are short by construction -- distance/mult of a sweep -- so
    it is worth checking they still get enough command points to be traced
    rather than approximated. At these values they get 15 and 25; below about 5
    the servo receives a shrug instead of a tick.
    """

    amplitude: float = 0.15          # cursor magnitude of the sweep
    sweep_s: float = 1.75             # extreme to extreme
    retreat: float = 0.30            # how far back it pulls, as a fraction
    overshoot: float = 0.20          # how far past the extreme it then goes
    # Separate multipliers so the two halves of the flourish can differ. Equal
    # values give one continuous gesture at a constant pace; a higher
    # overshoot_mult makes the retreat a wind-up that the overshoot cracks
    # through, which reads as a snap rather than a sway.
    retreat_mult: float = 2.0        # retreat speed, x the sweep's pace
    overshoot_mult: float = 3.0      # overshoot speed, x the sweep's pace
    cycles: int = 2                  # per invocation; bounds reaction lag
    direction_deg: float = 60.0      # perpendicular to motor 2: left/right
    time_per_point: float = 0.02


SIDE_SIDE_CONFIG = SideSideConfig()
# EXCITED: the same shape with more energy. Wider, quicker, and a gentler
# multiplier so the beat keeps enough points to read at the shorter sweep.
SIDE_SIDE_FAST_CONFIG = SideSideConfig(amplitude=0.18, sweep_s=0.8,
                                       retreat_mult=2.0,
                                       overshoot_mult=2.0)
GRAB_CONFIG = GrabMotionConfig()
RELEASE_CONFIG = ReleaseMotionConfig()
HIGH_FIVE_CONFIG = HighFiveMotionConfig()


def perform_yes_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    *,
    noise_scale: float = 0.0,
) -> None:
    """Perform yes/nodding motion."""
    for _ in range(4):
        target_positions_down, _ = cursor_to_motor_positions(
            cursor_pos=YES_CONFIG.down_position,
            calibrated_ticks_map=calibrated_ticks_map,
            noise_scale=noise_scale,
        )
        motor_controller.set_positions(target_positions_down)
        time.sleep(YES_CONFIG.hold_duration)

        target_positions_centre, _ = cursor_to_motor_positions(
            cursor_pos=YES_CONFIG.center_position,
            calibrated_ticks_map=calibrated_ticks_map,
            noise_scale=noise_scale,
        )
        motor_controller.set_positions(target_positions_centre)
        time.sleep(YES_CONFIG.hold_duration)


def perform_no_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    *,
    noise_scale: float = 0.0,
) -> None:
    """Perform no/head-shake motion."""
    target_positions_down, _ = cursor_to_motor_positions(
        cursor_pos=NO_CONFIG.down_position,
        calibrated_ticks_map=calibrated_ticks_map,
        noise_scale=noise_scale,
    )
    motor_controller.set_positions(target_positions_down)
    time.sleep(NO_CONFIG.initial_delay)

    for _ in range(4):
        target_positions_left, _ = cursor_to_motor_positions(
            cursor_pos=NO_CONFIG.left_position,
            calibrated_ticks_map=calibrated_ticks_map,
            noise_scale=noise_scale,
        )
        motor_controller.set_positions(target_positions_left)
        time.sleep(NO_CONFIG.hold_duration)

        target_positions_right, _ = cursor_to_motor_positions(
            cursor_pos=NO_CONFIG.right_position,
            calibrated_ticks_map=calibrated_ticks_map,
            noise_scale=noise_scale,
        )
        motor_controller.set_positions(target_positions_right)
        time.sleep(NO_CONFIG.hold_duration)


def perform_shake_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    *,
    noise_scale: float = 0.0,
) -> None:
    """Perform shake motion."""
    for _ in range(4):
        target_positions_left, _ = cursor_to_motor_positions(
            cursor_pos=SHAKE_CONFIG.left_position,
            calibrated_ticks_map=calibrated_ticks_map,
            noise_scale=noise_scale,
        )
        motor_controller.set_positions(target_positions_left)
        time.sleep(SHAKE_CONFIG.hold_duration)

        target_positions_right, _ = cursor_to_motor_positions(
            cursor_pos=SHAKE_CONFIG.right_position,
            calibrated_ticks_map=calibrated_ticks_map,
            noise_scale=noise_scale,
        )
        motor_controller.set_positions(target_positions_right)
        time.sleep(SHAKE_CONFIG.hold_duration)


def perform_circle_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    *,
    noise_scale: float = 0.0,
) -> None:
    """Perform circular motion in XY plane."""
    for _ in range(4):
        for i in range(CIRCLE_CONFIG.points_per_circle):
            angle = (i / CIRCLE_CONFIG.points_per_circle) * 2 * np.pi
            cursor_pos = np.array(
                [
                    CIRCLE_CONFIG.radius * np.cos(angle),
                    CIRCLE_CONFIG.radius * np.sin(angle),
                ]
            )

            target_positions, _ = cursor_to_motor_positions(
                cursor_pos=cursor_pos,
                calibrated_ticks_map=calibrated_ticks_map,
                noise_scale=noise_scale,
            )
            motor_controller.set_positions(target_positions)
            time.sleep(CIRCLE_CONFIG.time_per_point)


def perform_slow_circle_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    *,
    noise_scale: float = 0.0,
    should_stop=None,
) -> None:
    """Circular motion at a rate the servos can actually track.

    Differs from perform_circle_motion in three ways, each addressing a measured
    problem rather than a preference: more points (smaller steps), a longer
    dwell (lower rate), and an eased entry and exit (no slam from neutral).
    """
    cfg = SLOW_CIRCLE_CONFIG
    step = 2 * np.pi / cfg.points_per_circle   # angular advance per command

    def go(angle, radius):
        target_positions, _ = cursor_to_motor_positions(
            cursor_pos=np.array([radius * np.cos(angle), radius * np.sin(angle)],
                                dtype=float),
            calibrated_ticks_map=calibrated_ticks_map,
            noise_scale=noise_scale,
        )
        motor_controller.set_positions(target_positions)
        time.sleep(cfg.time_per_point)

    angle = 0.0

    # Spiral OUT: advance the angle while growing the radius, so the tentacle
    # arrives on the circle already moving along it.
    for k in range(cfg.entry_steps):
        f = k / cfg.entry_steps
        go(angle, cfg.min_radius + (cfg.radius - cfg.min_radius) * f)
        angle += step

    for _ in range(cfg.revolutions):
        for _i in range(cfg.points_per_circle):
            go(angle, cfg.radius)
            angle += step

    # Spiral IN, mirroring the entry, ending just inside the deadzone.
    for k in range(cfg.entry_steps):
        f = 1.0 - (k + 1) / cfg.entry_steps
        go(angle, cfg.min_radius + (cfg.radius - cfg.min_radius) * f)
        angle += step


def perform_wave_packet_am_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    *,
    noise_scale: float = 0.0,
) -> None:
    """Amplitude-modulated Gaussian wave packet, repeated `repeats` times.

    The phase accumulates across repeats instead of resetting, so rotation
    continues in one direction through every seam. The envelope is what
    restarts, and since it is ~0 at both ends of a packet the position is
    continuous there as well.
    """
    cfg = PACKET_AM_CONFIG
    phase0 = np.radians(cfg.direction_deg)
    d = np.array([np.cos(phase0), np.sin(phase0)])
    half = cfg.span_sigmas * cfg.sigma_s / 2.0
    n = max(2, int(round(2 * half / cfg.time_per_point)))

    theta = 0.0
    for rep in range(max(1, cfg.repeats)):
        # Skip the first sample of later repeats: it duplicates the last sample
        # of the previous packet, both sitting at envelope ~0.
        for i in range(0 if rep == 0 else 1, n + 1):
            t = -half + i * cfg.time_per_point
            envelope = np.exp(-0.5 * (t / cfg.sigma_s) ** 2)
            theta += 2 * np.pi * cfg.frequency_hz * cfg.time_per_point
            if cfg.circular:
                # Quadrature: cos and sin are 90 deg apart, so the cursor turns.
                cursor = cfg.amplitude * envelope * np.array(
                    [np.cos(theta + phase0), np.sin(theta + phase0)]
                )
            else:
                cursor = cfg.amplitude * envelope * np.sin(theta) * d
            target_positions, _ = cursor_to_motor_positions(
                cursor_pos=cursor,
                calibrated_ticks_map=calibrated_ticks_map,
                noise_scale=noise_scale,
                # The default 0.01 deadzone stops jitter around centre in
                # interactive control; for a scripted waveform it is distortion.
                # This packet passes through ~0 at every seam, and 34% of the
                # linear form's commands were being snapped flat by it.
                cursor_deadzone=1e-6,
            )
            motor_controller.set_positions(target_positions)
            time.sleep(cfg.time_per_point)


def perform_wave_packet_fm_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    *,
    noise_scale: float = 0.0,
) -> None:
    """Frequency-modulated Gaussian wave packet, repeated `repeats` times.

    Constant-rate rotation is what changes: f(t) is a Gaussian, the phase is
    INTEGRATED step by step, and the radius follows the configured envelope.
    Phase accumulates across repeats so the seams are continuous.
    """
    cfg = PACKET_FM_CONFIG
    half = cfg.span_sigmas * cfg.sigma_s / 2.0
    n = max(2, int(round(2 * half / cfg.time_per_point)))

    def go(angle, radius):
        target_positions, _ = cursor_to_motor_positions(
            cursor_pos=np.array([radius * np.cos(angle), radius * np.sin(angle)]),
            calibrated_ticks_map=calibrated_ticks_map,
            noise_scale=noise_scale,
            # The triangle envelope takes the radius to ~0, well inside the
            # default 0.01 deadzone.
            cursor_deadzone=1e-6,
        )
        motor_controller.set_positions(target_positions)
        time.sleep(cfg.time_per_point)

    theta = 0.0
    for rep in range(max(1, cfg.repeats)):
        for i in range(0 if rep == 0 else 1, n + 1):
            t = -half + i * cfg.time_per_point
            f = cfg.f_min_hz + (cfg.f_max_hz - cfg.f_min_hz) * np.exp(
                -0.5 * (t / cfg.sigma_s) ** 2
            )
            # Integrate the phase; evaluating 2*pi*f(t)*t instead would be a
            # different signal and would jump wherever f changes.
            theta += 2 * np.pi * f * cfg.time_per_point

            if cfg.envelope == "triangle":
                # Linear ramp: 0 at both ends, full radius at the midpoint,
                # where the frequency also peaks.
                radius = cfg.radius * max(0.0, 1.0 - abs(t) / half)
            else:
                if i < cfg.entry_steps:
                    frac = i / cfg.entry_steps
                elif i > n - cfg.entry_steps:
                    frac = (n - i) / cfg.entry_steps
                else:
                    frac = 1.0
                radius = cfg.min_radius + (cfg.radius - cfg.min_radius) * frac
            go(theta, radius)


def perform_slow_breathe_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    *,
    noise_scale: float = 0.0,
    should_stop=None,
) -> None:
    """Slow undulating grab: a unipolar swell along motor 2's axis, repeated."""
    _breathe(motor_controller, calibrated_ticks_map, SLOW_BREATHE_CONFIG,
             noise_scale=noise_scale, should_stop=should_stop)


def perform_normal_breathe_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    *,
    noise_scale: float = 0.0,
    should_stop=None,
) -> None:
    """The same breath at double frequency. See NORMAL_BREATHE_CONFIG."""
    _breathe(motor_controller, calibrated_ticks_map, NORMAL_BREATHE_CONFIG,
             noise_scale=noise_scale, should_stop=should_stop)


def _breathe(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    cfg: SlowBreatheConfig,
    *,
    noise_scale: float = 0.0,
    should_stop=None,
) -> None:
    d = np.array([np.cos(np.radians(cfg.direction_deg)),
                  np.sin(np.radians(cfg.direction_deg))])
    peak_magnitude = cfg.peak_ticks / float(MOTOR_ONE_FULL_TURN_TICKS)
    per_cycle = max(2, int(round(cfg.period_s / cfg.time_per_point)))

    for rep in range(max(1, cfg.repeats)):
        # Skip index 0 on later cycles: it repeats the previous cycle's final
        # sample, both sitting at the bottom of the breath.
        for i in range(0 if rep == 0 else 1, per_cycle + 1):
            phase = 2 * np.pi * i / per_cycle
            magnitude = peak_magnitude * (1.0 - np.cos(phase)) / 2.0
            target_positions, _ = cursor_to_motor_positions(
                cursor_pos=magnitude * d,
                calibrated_ticks_map=calibrated_ticks_map,
                noise_scale=noise_scale,
                # The breath returns to zero every cycle, well inside the
                # default 0.01 deadzone, which would otherwise snap the bottom
                # of each breath flat to the calibrated position.
                cursor_deadzone=1e-6,
            )
            motor_controller.set_positions(target_positions)
            time.sleep(cfg.time_per_point)
            if _stopping(should_stop):
                _ease_to_neutral(motor_controller, calibrated_ticks_map,
                                 magnitude * d)
                return


# Grab is the deepest reach in the set and the closest to the encoder end stop:
# at the 2026-09-04 zeros it commanded 4014 of 4095, 81 ticks from refusal. It is
# also the one whose margin shrinks on its own, because retension only ever winds
# the zeros UP and motor 2 -- the axis grab drives, alignment +1.0 -- has the
# least headroom.
#
# So the depth is DERIVED from the live calibration rather than fixed. A literal
# that is correct today stops being correct the first time the tendons are
# retensioned, silently.
GRAB_TICK_MARGIN = 100      # never command within this of 0 or 4095
GRAB_MIN_TICKS = 500        # below this the grip is too shallow to be useful


def max_grab_magnitude(calibrated_ticks_map: Dict[str, int],
                       margin: int = GRAB_TICK_MARGIN) -> float:
    """Deepest grab that keeps every motor `margin` ticks inside 0..4095.

    Offset for a motor is alignment * magnitude * 4096, so each motor caps the
    magnitude at (limit - zero) / (alignment * 4096); the smallest cap wins.
    Bounds both ends: motors anti-aligned with the grab axis travel DOWNWARD,
    and which motor binds can change as the zeros move.
    """
    direction = np.asarray(GRAB_CONFIG.grab_cursor_pos, float)
    n = np.linalg.norm(direction)
    if n < 1e-9:
        return 0.0
    direction = direction / n

    caps = []
    for name, pos in MOTOR_NORMALIZED_POSITIONS.items():
        pos = np.asarray(pos, float)
        pn = np.linalg.norm(pos)
        if pn < 1e-9:
            continue
        align = float(np.dot(direction, pos / pn))
        if abs(align) < 1e-6:
            continue
        zero = calibrated_ticks_map.get(name, 0)
        limit = (MOTOR_ONE_FULL_TURN_TICKS - 1 - margin) if align > 0 else margin
        caps.append((limit - zero) / (align * MOTOR_ONE_FULL_TURN_TICKS))
    return max(0.0, min(caps)) if caps else 0.0


# How long an interrupted primitive takes to ease back to neutral. Bailing out
# of a loop leaves the tentacle wherever it was; execute_behavior's reset would
# then snap it home in one command, which is the jump the arch already had to be
# fixed for. 0.3 s at 700 ticks is 2300 ticks/s, well under the ceiling.
INTERRUPT_EASE_S = 0.30


def _stopping(should_stop) -> bool:
    return should_stop is not None and should_stop()


def _ease_to_neutral(motor_controller, calibrated_ticks_map, cursor,
                     seconds: float = INTERRUPT_EASE_S, dt: float = 0.02) -> None:
    """Ramp a cursor back to zero, for a primitive cut short mid-motion."""
    cursor = np.asarray(cursor, float)
    if float(np.linalg.norm(cursor)) < 1e-6:
        return
    n = max(2, int(round(seconds / dt)))
    for i in range(n - 1, -1, -1):
        ease = (1.0 - np.cos(np.pi * i / n)) / 2.0
        target_positions, _ = cursor_to_motor_positions(
            cursor_pos=cursor * ease,
            calibrated_ticks_map=calibrated_ticks_map,
            cursor_deadzone=1e-6,
        )
        motor_controller.set_positions(target_positions)
        time.sleep(dt)


def side_side_segments(cfg: SideSideConfig):
    """(target fraction, duration) from neutral, tick-tock, back to neutral.

    Both beat durations are distance/speed_mult scaled by the sweep, so the
    retreat and the overshoot travel at the same pace even though the overshoot
    covers further ground.
    """
    t_ret = cfg.sweep_s * cfg.retreat / cfg.retreat_mult
    t_over = cfg.sweep_s * (cfg.retreat + cfg.overshoot) / cfg.overshoot_mult
    segs = [(+1.0, cfg.sweep_s / 2)]                      # lead in from centre
    for i in range(max(1, cfg.cycles)):
        segs += [(+1.0 - cfg.retreat, t_ret), (+1.0 + cfg.overshoot, t_over)]
        segs += [(-1.0, cfg.sweep_s)]
        segs += [(-1.0 + cfg.retreat, t_ret), (-1.0 - cfg.overshoot, t_over)]
        segs += ([(0.0, cfg.sweep_s / 2)] if i == cfg.cycles - 1
                 else [(+1.0, cfg.sweep_s)])
    return segs, (t_ret, t_over)


def perform_side_side_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    cfg: SideSideConfig = None,
    *,
    noise_scale: float = 0.0,
    should_stop=None,
) -> None:
    """Play the tick-tock sway described by `cfg`."""
    cfg = cfg or SIDE_SIDE_CONFIG
    segs, _ = side_side_segments(cfg)
    d = np.array([np.cos(np.radians(cfg.direction_deg)),
                  np.sin(np.radians(cfg.direction_deg))])
    cur = 0.0
    for target, dur in segs:
        n = max(2, int(round(dur / cfg.time_per_point)))
        for i in range(1, n + 1):
            # raised cosine: zero velocity at both ends of every segment, so
            # the joins cannot jerk -- and this reverses direction twice at
            # each extreme, which is exactly where a corner would show
            ease = (1.0 - np.cos(np.pi * i / n)) / 2.0
            frac = cur + (target - cur) * ease
            target_positions, _ = cursor_to_motor_positions(
                cursor_pos=d * cfg.amplitude * frac,
                calibrated_ticks_map=calibrated_ticks_map,
                noise_scale=noise_scale,
                # the sweep passes through zero twice a cycle, well inside the
                # default 0.01 deadzone, which would flatten the middle of every
                # traverse to the calibrated pose
                cursor_deadzone=1e-6,
            )
            motor_controller.set_positions(target_positions)
            time.sleep(cfg.time_per_point)
            if _stopping(should_stop):
                _ease_to_neutral(motor_controller, calibrated_ticks_map,
                                 d * cfg.amplitude * frac)
                return
        cur = target


def perform_arched_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    noise_scale: float = 0.0,
) -> None:
    """Ramp linearly to the arch offset and hold there."""
    cfg = ARCHED_CONFIG
    d = np.array([np.cos(np.radians(cfg.direction_deg)),
                  np.sin(np.radians(cfg.direction_deg))])
    peak = cfg.offset_ticks / float(MOTOR_ONE_FULL_TURN_TICKS)
    steps = max(2, int(round(cfg.ramp_s / cfg.time_per_point)))
    logger.info("Arching to %d ticks over %.1fs", cfg.offset_ticks, cfg.ramp_s)
    for i in range(1, steps + 1):
        target_positions, _ = cursor_to_motor_positions(
            cursor_pos=d * (peak * i / steps),
            calibrated_ticks_map=calibrated_ticks_map,
            noise_scale=noise_scale,
            # the ramp starts at zero, well inside the default deadzone, which
            # would otherwise flatten the first third of the motion
            cursor_deadzone=1e-6,
        )
        motor_controller.set_positions(target_positions)
        time.sleep(cfg.time_per_point)


def perform_unarch_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    noise_scale: float = 0.0,
) -> None:
    """Ramp back from the arch to neutral, mirroring perform_arched_motion.

    The arch takes 1.5 s to reach its pose and release_object put it back in a
    single command -- 699 ticks in one step, whose effective rate depends only
    on when the next command happens to land. The static check flagged it at 15x
    the servo ceiling, which is what a jump looks like when it is measured
    rather than assumed.

    ASSUMES IT STARTS FROM THE ARCH POSE, because it exists only as arched's
    exit. Called from anywhere else it would first snap out to the arch and then
    ramp back, which is worse than what it replaces.
    """
    cfg = ARCHED_CONFIG
    d = np.array([np.cos(np.radians(cfg.direction_deg)),
                  np.sin(np.radians(cfg.direction_deg))])
    peak = cfg.offset_ticks / float(MOTOR_ONE_FULL_TURN_TICKS)
    steps = max(2, int(round(cfg.ramp_s / cfg.time_per_point)))
    logger.info("Unarching from %d ticks over %.1fs", cfg.offset_ticks, cfg.ramp_s)
    for i in range(steps, -1, -1):
        target_positions, _ = cursor_to_motor_positions(
            cursor_pos=d * (peak * i / steps),
            calibrated_ticks_map=calibrated_ticks_map,
            noise_scale=noise_scale,
            cursor_deadzone=1e-6,
        )
        motor_controller.set_positions(target_positions)
        time.sleep(cfg.time_per_point)


def perform_grab_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    noise_scale: float = 0.0,
) -> None:
    """Move tentacle to a grab pose, no deeper than the calibration allows."""
    wanted = float(np.linalg.norm(GRAB_CONFIG.grab_cursor_pos))
    cap = max_grab_magnitude(calibrated_ticks_map)
    magnitude = min(wanted, cap)
    ticks = int(round(magnitude * MOTOR_ONE_FULL_TURN_TICKS))

    if cap < wanted:
        logger.warning(
            "Grab depth capped %.4f -> %.4f (%d ticks) to keep %d ticks clear "
            "of the encoder range at the current zeros %s",
            wanted, cap, ticks, GRAB_TICK_MARGIN, calibrated_ticks_map)
    if ticks < GRAB_MIN_TICKS:
        logger.warning(
            "GRAB IS NOW ONLY %d TICKS DEEP (floor %d). The zeros have drifted "
            "far enough that the grip is not worth having -- re-zero the motors "
            "rather than relying on this. Current zeros: %s",
            ticks, GRAB_MIN_TICKS, calibrated_ticks_map)

    direction = np.asarray(GRAB_CONFIG.grab_cursor_pos, float)
    direction = direction / (np.linalg.norm(direction) or 1.0)
    cursor = direction * magnitude
    logger.info("Moving to GRAB position: %s (magnitude %.4f)", cursor, magnitude)
    target_positions_grab, _ = cursor_to_motor_positions(
        cursor_pos=cursor,
        calibrated_ticks_map=calibrated_ticks_map,
        noise_scale=noise_scale,
    )
    motor_controller.set_positions(target_positions_grab)
    time.sleep(GRAB_CONFIG.hold_duration)


def perform_release_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    noise_scale: float = 0.0,
) -> None:
    """Move tentacle to neutral position, releasing grab."""
    logger.info("Moving to NEUTRAL position: %s", RELEASE_CONFIG.neutral_cursor_pos)
    target_positions_neutral, _ = cursor_to_motor_positions(
        cursor_pos=RELEASE_CONFIG.neutral_cursor_pos,
        calibrated_ticks_map=calibrated_ticks_map,
        noise_scale=noise_scale,
    )
    motor_controller.set_positions(target_positions_neutral)
    time.sleep(RELEASE_CONFIG.hold_duration)


def perform_high_five_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    *,
    noise_scale: float = 0.0,
) -> None:
    """Perform high five motion."""
    # Move to high five position
    target_positions_high_five, _ = cursor_to_motor_positions(
        cursor_pos=HIGH_FIVE_CONFIG.high_five_position,
        calibrated_ticks_map=calibrated_ticks_map,
        noise_scale=noise_scale,
    )
    motor_controller.set_positions(target_positions_high_five)
    time.sleep(HIGH_FIVE_CONFIG.hold_duration)

    # Return to center position
    target_positions_centre, _ = cursor_to_motor_positions(
        cursor_pos=HIGH_FIVE_CONFIG.center_position,
        calibrated_ticks_map=calibrated_ticks_map,
        noise_scale=noise_scale,
    )
    motor_controller.set_positions(target_positions_centre)
    time.sleep(HIGH_FIVE_CONFIG.hold_duration)


def execute_behavior(
    motor_controller: MotorController,
    behavior: MotionBehavior,
    *,
    noise_scale: float = 0.010,
    should_stop=None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute a motion behavior primitive.

    Args:
        motor_controller: Connected motor controller
        behavior: The motion behavior to execute
        noise_scale: Scale of random noise to apply
        **kwargs: Additional behavior-specific parameters

    Returns:
        Dictionary with execution result information
    """
    if not motor_controller.is_connected:
        return {
            "behavior": behavior.value,
            "status": "error",
            "message": "Motor controller not connected",
        }

    # Get calibration data
    calibrated_ticks_map = motor_controller.get_calibration_data()

    behaviors_performed = False
    reset_after_sequence = False

    try:
        if behavior == MotionBehavior.YES:
            perform_yes_motion(
                motor_controller,
                calibrated_ticks_map,
                noise_scale=noise_scale,
            )
            behaviors_performed = True
            reset_after_sequence = True

        elif behavior == MotionBehavior.NO:
            perform_no_motion(
                motor_controller,
                calibrated_ticks_map,
                noise_scale=noise_scale,
            )
            behaviors_performed = True
            reset_after_sequence = True

        elif behavior == MotionBehavior.SHAKE:
            perform_shake_motion(
                motor_controller,
                calibrated_ticks_map,
                noise_scale=noise_scale,
            )
            behaviors_performed = True
            reset_after_sequence = True

        elif behavior == MotionBehavior.CIRCLE:
            perform_circle_motion(
                motor_controller,
                calibrated_ticks_map,
                noise_scale=noise_scale,
            )
            behaviors_performed = True
            reset_after_sequence = True

        elif behavior == MotionBehavior.UNARCH:
            perform_unarch_motion(
                motor_controller, calibrated_ticks_map, noise_scale=noise_scale,
            )
            behaviors_performed = True
            reset_after_sequence = True

        elif behavior == MotionBehavior.SIDE_SIDE:
            perform_side_side_motion(
                motor_controller, calibrated_ticks_map,
                SIDE_SIDE_CONFIG, noise_scale=noise_scale,
                should_stop=should_stop,
            )
            behaviors_performed = True
            reset_after_sequence = True

        elif behavior == MotionBehavior.SIDE_SIDE_FAST:
            perform_side_side_motion(
                motor_controller, calibrated_ticks_map,
                SIDE_SIDE_FAST_CONFIG, noise_scale=noise_scale,
                should_stop=should_stop,
            )
            behaviors_performed = True
            reset_after_sequence = True

        elif behavior == MotionBehavior.ARCHED:
            perform_arched_motion(
                motor_controller,
                calibrated_ticks_map,
                noise_scale=noise_scale,
            )
            behaviors_performed = True
            reset_after_sequence = False       # a held posture, like grab

        elif behavior == MotionBehavior.NORMAL_BREATHE:
            perform_normal_breathe_motion(
                motor_controller,
                calibrated_ticks_map,
                noise_scale=noise_scale,
                should_stop=should_stop,
            )
            behaviors_performed = True
            reset_after_sequence = True

        elif behavior == MotionBehavior.SLOW_BREATHE:
            perform_slow_breathe_motion(
                motor_controller, calibrated_ticks_map, noise_scale=noise_scale,
            )
            behaviors_performed = True
            reset_after_sequence = True

        elif behavior == MotionBehavior.PACKET_AM:
            perform_wave_packet_am_motion(
                motor_controller, calibrated_ticks_map, noise_scale=noise_scale,
            )
            behaviors_performed = True
            reset_after_sequence = True

        elif behavior == MotionBehavior.PACKET_FM:
            perform_wave_packet_fm_motion(
                motor_controller, calibrated_ticks_map, noise_scale=noise_scale,
            )
            behaviors_performed = True
            reset_after_sequence = True

        elif behavior == MotionBehavior.SLOW_CIRCLE:
            perform_slow_circle_motion(
                motor_controller,
                calibrated_ticks_map,
                noise_scale=noise_scale,
                should_stop=should_stop,
            )
            behaviors_performed = True
            reset_after_sequence = True

        elif behavior == MotionBehavior.GRAB:
            perform_grab_motion(
                motor_controller,
                calibrated_ticks_map,
                noise_scale=noise_scale,
            )
            behaviors_performed = True
            reset_after_sequence = False

        elif behavior == MotionBehavior.RELEASE:
            perform_release_motion(
                motor_controller,
                calibrated_ticks_map,
                noise_scale=noise_scale,
            )
            behaviors_performed = True
            reset_after_sequence = True

        elif behavior == MotionBehavior.HIGH_FIVE:
            perform_high_five_motion(
                motor_controller,
                calibrated_ticks_map,
                noise_scale=noise_scale,
            )
            behaviors_performed = True
            reset_after_sequence = True

        # Handle reset logic
        if behaviors_performed and reset_after_sequence:
            logger.info("Behaviors complete. Resetting motors to zero")
            motor_controller.reset_to_calibrated_positions()
            time.sleep(0.2)
        elif behaviors_performed and not reset_after_sequence:
            logger.info("Grab behavior complete. Motors will remain in grab pose")

        return {
            "behavior": behavior.value,
            "status": "success",
            "message": f"Successfully executed {behavior.value} behavior",
            "reset_performed": reset_after_sequence,
        }

    except Exception as e:
        logger.error("Error executing behavior %s: %s", behavior.value, e)
        return {
            "behavior": behavior.value,
            "status": "error",
            "message": f"Error executing behavior: {e}",
        }


@app.command()
def run(
    behavior: str = typer.Argument(
        help="Motion behavior to test: yes, no, shake, circle, grab, release, high_five"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    noise_scale: float = typer.Option(
        0.010, "--noise", "-n", help="Noise scale for motion randomization"
    ),
) -> None:
    """Test a motion primitive behavior."""

    console.print(f"[bold blue]Testing Motion Primitive: {behavior}[/bold blue]")

    try:
        # Validate behavior
        try:
            motion_behavior = MotionBehavior(behavior)
        except ValueError:
            console.print(f"[red]Error: Unknown behavior '{behavior}'[/red]")
            console.print("[yellow]Available behaviors:[/yellow]")
            for b in MotionBehavior:
                console.print(f"  • {b.value}")
            raise typer.Exit(1)

        # Create config from file
        hardware_config = get_hardware_config(config)

        console.print(
            f"Connecting to motors on port: [cyan]{hardware_config.port}[/cyan]"
        )

        # Connect to motors
        with console.status("[bold green]Connecting..."):
            motor_controller = MotorController(hardware_config)
            motor_controller.connect()

        console.print("[green]✓[/green] Connected to motors")

        console.print(f"[bold yellow]Executing {behavior} behavior...[/bold yellow]")

        result = execute_behavior(
            motor_controller=motor_controller,
            behavior=motion_behavior,
            noise_scale=noise_scale,
        )

        if result["status"] == "success":
            console.print(f"[green]✓[/green] {result['message']}")
            if result.get("reset_performed"):
                console.print("[dim]Motors reset to calibrated positions[/dim]")
        else:
            console.print(f"[red]Error: {result['message']}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)
