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
from ..common.constants import MOTOR_NORMALIZED_POSITIONS
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
    initial_delay: float = 0.05
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


def perform_grab_motion(
    motor_controller: MotorController,
    calibrated_ticks_map: Dict[str, int],
    noise_scale: float = 0.0,
) -> None:
    """Move tentacle to predefined grabbing position."""
    logger.info("Moving to GRAB position: %s", GRAB_CONFIG.grab_cursor_pos)
    target_positions_grab, _ = cursor_to_motor_positions(
        cursor_pos=GRAB_CONFIG.grab_cursor_pos,
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

        elif behavior == MotionBehavior.SLOW_CIRCLE:
            perform_slow_circle_motion(
                motor_controller,
                calibrated_ticks_map,
                noise_scale=noise_scale,
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
