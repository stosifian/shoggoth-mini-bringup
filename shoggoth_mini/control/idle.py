"""Idle motion with breathing pattern for natural robotic movement. Code is heavily vibe-coded and a refactor is warranted."""

import logging
import time
import math
import random
import threading
from typing import Dict, Optional, Tuple
import numpy as np
import typer
from rich.console import Console

from .geometry import cursor_to_motor_positions
from ..hardware.motors import MotorController
from ..configs import get_hardware_config

console = Console()
app = typer.Typer(help="Idle motion utilities")
logger = logging.getLogger(__name__)


class BreathingPattern:
    """Generates natural breathing-like motion patterns for robotic idle behavior.

    This class creates smooth, organic movement patterns that simulate natural breathing
    or swaying motions. The pattern oscillates in 2D space with natural variations in
    frequency, amplitude, and direction to avoid repetitive, mechanical-looking movement.

    Mathematical Model:
        cursor(t) = A(t) * sin(2π * f(t) * (t-t0) + φ) * direction(t)

        Where:
        - A(t): Amplitude that can vary over time and smoothly transition
        - f(t): Frequency that jitters randomly every jitter_interval seconds:
                f(t) = f0 * (1 + U[-jitter, +jitter])
        - direction(t): Unit vector that slowly changes direction over time
        - φ: Phase offset to maintain continuity during parameter changes

    Key Features:
        - Smooth sinusoidal base motion with configurable period
        - Random frequency jitter to break up repetitive patterns
        - Gradual direction changes for natural wandering motion
        - Amplitude variations within specified ranges
        - Smooth transitions between parameter changes to avoid jarring movements
        - Optional origin avoidance to maintain circular motion patterns

    The pattern generates 2D cursor positions in normalized coordinates [-1, 1] that
    can be converted to motor positions using geometry conversion functions.
    """

    def __init__(
        self,
        base_period: float = 8.0,
        jitter: float = 0.15,
        jitter_interval: float = 5.0,
        amplitude: float = 0.5,
        direction: np.ndarray = np.array([0.0, 1.0]),
        direction_jitter_interval: float = 10.0,
        direction_change_duration: float = 2.0,
        origin_avoidance_radius: float = 0.0,
        amplitude_range: Optional[Tuple[float, float]] = None,
        base_period_range: Optional[Tuple[float, float]] = None,
        direction_jitter_interval_range: Optional[Tuple[float, float]] = None,
        frequency_transition_duration: float = 0.5,
        jitter_frequency_transition_duration: float = 0.1,
    ):
        """Initialize breathing pattern."""
        self.t0 = time.time()

        self.initial_amplitude = amplitude
        self.initial_base_period = base_period

        self.amplitude_min = (
            amplitude_range[0] if amplitude_range else self.initial_amplitude
        )
        self.amplitude_max = (
            amplitude_range[1] if amplitude_range else self.initial_amplitude
        )
        self.base_period_min = (
            base_period_range[0] if base_period_range else self.initial_base_period
        )
        self.base_period_max = (
            base_period_range[1] if base_period_range else self.initial_base_period
        )

        self.amp_base = self.initial_amplitude
        self._previous_amp_base = self.initial_amplitude
        self.base_f = 1.0 / self.initial_base_period
        if self.base_f == 0:
            self.base_f = 1.0 / 1e-6

        self.jitter = jitter
        self.jitter_ivl = jitter_interval
        self.direction = direction / np.linalg.norm(direction)
        self._previous_direction = np.copy(self.direction)

        self.direction_jitter_interval_min = (
            direction_jitter_interval_range[0]
            if direction_jitter_interval_range
            else direction_jitter_interval
        )
        self.direction_jitter_interval_max = (
            direction_jitter_interval_range[1]
            if direction_jitter_interval_range
            else direction_jitter_interval
        )
        self.direction_jitter_interval = random.uniform(
            self.direction_jitter_interval_min, self.direction_jitter_interval_max
        )

        self.direction_change_duration = direction_change_duration
        self.origin_avoidance_radius = origin_avoidance_radius

        self._next_jitter_t = self.t0
        self._current_f = self.base_f * (
            1.0 + random.uniform(-self.jitter, +self.jitter)
        )
        self._phase0 = 0.0

        self._next_dynamic_param_resample_t = self.t0
        self._direction_change_start_t = self.t0

        self.frequency_transition_duration = frequency_transition_duration
        if hasattr(self, "_frequency_transition_start_t"):
            del self._frequency_transition_start_t

        self.jitter_frequency_transition_duration = jitter_frequency_transition_duration
        self._jitter_transition_active = False

    def _initiate_smooth_jitter_transition(self, t_change: float) -> None:
        """Set up parameters to start smooth transition for jittered frequency."""
        self._jitter_transition_active = True
        self._jitter_transition_start_t = t_change
        self._jitter_transition_f_initial = self._current_f
        self._jitter_transition_f_target = self.base_f * (
            1.0 + random.uniform(-self.jitter, +self.jitter)
        )
        self._jitter_transition_phase_at_start = (
            2 * math.pi * self._current_f * (t_change - self.t0) + self._phase0
        )

    def _perform_instantaneous_jitter_update(self, t_change: float) -> None:
        """Apply jittered frequency change instantaneously."""
        old_current_f = self._current_f
        self._current_f = self.base_f * (
            1.0 + random.uniform(-self.jitter, +self.jitter)
        )
        self._phase0 += (
            2 * math.pi * (old_current_f - self._current_f) * (t_change - self.t0)
        )

    def _resample_dynamic_parameters(self, t: float) -> None:
        """Resample dynamic parameters for variation."""
        self._previous_direction = np.copy(self.direction)
        angle = random.uniform(0, 2 * math.pi)
        self.direction = np.array([math.cos(angle), math.sin(angle)])
        self._direction_change_start_t = t

        self._previous_amp_base = self.amp_base
        self.amp_base = random.uniform(self.amplitude_min, self.amplitude_max)

        new_base_period = random.uniform(self.base_period_min, self.base_period_max)
        if new_base_period < 0.01:
            new_base_period = 0.01
        self.base_f = 1.0 / new_base_period

        if self.frequency_transition_duration > 1e-6:
            self._phase_angle_at_frequency_transition_start = (
                2 * math.pi * self._current_f * (t - self.t0) + self._phase0
            )
            self._old_current_f_for_transition = self._current_f
            self._new_target_current_f = self.base_f * (
                1.0 + random.uniform(-self.jitter, +self.jitter)
            )
            self._frequency_transition_start_t = t
        else:
            if hasattr(self, "_frequency_transition_start_t"):
                del self._frequency_transition_start_t
            self._perform_instantaneous_jitter_update(t)
            self._next_jitter_t = t + self.jitter_ivl

        self.direction_jitter_interval = random.uniform(
            self.direction_jitter_interval_min, self.direction_jitter_interval_max
        )

    def __call__(self, t: float) -> np.ndarray:
        """Calculate cursor position at time t."""
        if t >= self._next_dynamic_param_resample_t:
            self._resample_dynamic_parameters(t)
            self._next_dynamic_param_resample_t = t + self.direction_jitter_interval

        effective_direction = self.direction
        effective_amp_base = self.amp_base

        if self.direction_change_duration > 0:
            elapsed_in_amp_dir_transition = t - self._direction_change_start_t
            if elapsed_in_amp_dir_transition < self.direction_change_duration:
                alpha = elapsed_in_amp_dir_transition / self.direction_change_duration

                prev_dir = self._previous_direction
                target_dir = self.direction
                dot_product = np.dot(prev_dir, target_dir)

                if dot_product < -0.9999:
                    angle_to_rotate = alpha * math.pi
                    cos_a = math.cos(angle_to_rotate)
                    sin_a = math.sin(angle_to_rotate)

                    px, py = prev_dir[0], prev_dir[1]
                    eff_dir_x = px * cos_a - py * sin_a
                    eff_dir_y = px * sin_a + py * cos_a
                    effective_direction = np.array([eff_dir_x, eff_dir_y])
                else:
                    interp_direction = (1 - alpha) * prev_dir + alpha * target_dir
                    norm = np.linalg.norm(interp_direction)
                    if norm > 1e-7:
                        effective_direction = interp_direction / norm
                    else:
                        effective_direction = target_dir

                effective_amp_base = (
                    1 - alpha
                ) * self._previous_amp_base + alpha * self.amp_base
            else:
                self._previous_direction = np.copy(self.direction)
                self._previous_amp_base = self.amp_base
                effective_direction = self.direction
                effective_amp_base = self.amp_base

        amp = effective_amp_base

        current_sine_argument = 0.0
        freq_path_handled_this_tick = False

        if t >= self._next_jitter_t:
            can_start_jitter = True
            if (
                hasattr(self, "_frequency_transition_start_t")
                and self.frequency_transition_duration > 1e-6
            ):
                can_start_jitter = False
            if (
                self._jitter_transition_active
                and self.jitter_frequency_transition_duration > 1e-6
            ):
                can_start_jitter = False

            if can_start_jitter:
                if self.jitter_frequency_transition_duration > 1e-6:
                    self._initiate_smooth_jitter_transition(t)
                else:
                    self._perform_instantaneous_jitter_update(t)
                    self._next_jitter_t = t + self.jitter_ivl

        if (
            hasattr(self, "_frequency_transition_start_t")
            and self.frequency_transition_duration > 1e-6
        ):
            freq_path_handled_this_tick = True
            elapsed_in_major_freq_trans = t - self._frequency_transition_start_t
            duration_major_trans = self.frequency_transition_duration
            if 0 <= elapsed_in_major_freq_trans < duration_major_trans:
                f_old = self._old_current_f_for_transition
                f_target = self._new_target_current_f
                integrated_freq_component = f_old * elapsed_in_major_freq_trans + (
                    f_target - f_old
                ) * (elapsed_in_major_freq_trans**2) / (2.0 * duration_major_trans)
                current_sine_argument = (
                    self._phase_angle_at_frequency_transition_start
                    + 2 * math.pi * integrated_freq_component
                )
            elif elapsed_in_major_freq_trans >= duration_major_trans:
                self._current_f = self._new_target_current_f
                f_old = self._old_current_f_for_transition
                integrated_freq_component_at_end = (
                    f_old * duration_major_trans
                    + (self._new_target_current_f - f_old) * duration_major_trans / 2.0
                )
                final_total_phase_angle = (
                    self._phase_angle_at_frequency_transition_start
                    + 2 * math.pi * integrated_freq_component_at_end
                )
                t_at_transition_end = (
                    self._frequency_transition_start_t + duration_major_trans
                )
                self._phase0 = final_total_phase_angle - (
                    2 * math.pi * self._current_f * (t_at_transition_end - self.t0)
                )
                del self._frequency_transition_start_t
                self._next_jitter_t = t + self.jitter_ivl
                current_sine_argument = final_total_phase_angle

        elif (
            self._jitter_transition_active
            and self.jitter_frequency_transition_duration > 1e-6
        ):
            freq_path_handled_this_tick = True
            elapsed_in_jitter_trans = t - self._jitter_transition_start_t
            duration_jitter_trans = self.jitter_frequency_transition_duration
            f_initial = self._jitter_transition_f_initial
            f_target = self._jitter_transition_f_target
            if 0 <= elapsed_in_jitter_trans < duration_jitter_trans:
                integrated_freq_component = f_initial * elapsed_in_jitter_trans + (
                    f_target - f_initial
                ) * (elapsed_in_jitter_trans**2) / (2.0 * duration_jitter_trans)
                current_sine_argument = (
                    self._jitter_transition_phase_at_start
                    + 2 * math.pi * integrated_freq_component
                )
            elif elapsed_in_jitter_trans >= duration_jitter_trans:
                self._current_f = f_target
                integrated_freq_component_at_end = (
                    f_initial * duration_jitter_trans
                    + (f_target - f_initial) * duration_jitter_trans / 2.0
                )
                final_total_phase_angle = (
                    self._jitter_transition_phase_at_start
                    + 2 * math.pi * integrated_freq_component_at_end
                )
                t_at_jitter_transition_end = (
                    self._jitter_transition_start_t + duration_jitter_trans
                )
                self._phase0 = final_total_phase_angle - (
                    2
                    * math.pi
                    * self._current_f
                    * (t_at_jitter_transition_end - self.t0)
                )
                current_sine_argument = final_total_phase_angle
                self._jitter_transition_active = False
                if hasattr(self, "_jitter_transition_start_t"):
                    del self._jitter_transition_start_t
                self._next_jitter_t = t + self.jitter_ivl

        if not freq_path_handled_this_tick:
            current_sine_argument = (
                2 * math.pi * self._current_f * (t - self.t0) + self._phase0
            )

        value = amp * math.sin(current_sine_argument)

        if self.origin_avoidance_radius > 1e-6:
            dx, dy = effective_direction[0], effective_direction[1]
            D_perp = np.array([-dy, dx])
            offset_component = self.origin_avoidance_radius * D_perp
            oscillation_component = value * effective_direction
            return offset_component + oscillation_component
        else:
            return value * effective_direction


class IdleMotionLoop:
    """Idle motion controller using breathing pattern."""

    def __init__(
        self,
        motor_controller: MotorController,
        pattern_config: Optional[Dict] = None,
        hz: float = 50.0,
        max_motor_step_per_loop: int = 30,
    ):
        """Initialize idle motion loop.

        Args:
            motor_controller: Connected motor controller
            pattern_config: Configuration dict for BreathingPattern
            hz: Update frequency in Hz
            max_motor_step_per_loop: Maximum motor step size per loop iteration
        """
        self.motor_controller = motor_controller
        self.pattern_config = pattern_config or {}
        self.hz = hz
        self.max_motor_step_per_loop = max_motor_step_per_loop

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pattern: Optional[BreathingPattern] = None
        self._current_motor_positions: Dict[str, int] = {}

    def start(self) -> None:
        """Start the idle motion loop in a background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Idle motion loop already running")
            return

        if not self.motor_controller.is_connected:
            raise RuntimeError(
                "Motor controller must be connected to start idle motion"
            )

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self, reset_to_calibrated: bool = True) -> None:
        """Stop the idle motion loop.

        Args:
            reset_to_calibrated: Whether to reset motors to calibrated positions
        """
        if not self._thread or not self._thread.is_alive():
            return

        self._stop_event.set()
        self._thread.join(timeout=2.0)

        if reset_to_calibrated and self.motor_controller.is_connected:
            logger.info("Resetting motors to calibrated positions")
            self.motor_controller.reset_to_calibrated_positions()

    @property
    def is_running(self) -> bool:
        """Check if the idle motion loop is currently running."""
        return self._thread is not None and self._thread.is_alive()

    def _run_loop(self) -> None:
        """Main idle motion loop."""
        try:
            # Get calibration data
            calibrated_ticks_map = self.motor_controller.get_calibration_data()
            motor_keys = list(calibrated_ticks_map.keys())

            # Initialize current motor positions to calibrated values
            self._current_motor_positions = calibrated_ticks_map.copy()

            logger.info("Idle Motion: Initializing motors to calibrated zero positions")

            # Reset motors to calibrated positions
            self.motor_controller.reset_to_calibrated_positions()
            time.sleep(0.5)

            logger.info("Idle Motion: Motors initialized")

            # Create breathing pattern
            self._pattern = BreathingPattern(**self.pattern_config)
            dt_loop = 1.0 / self.hz

            logger.info("Idle Motion: Starting breathing pattern loop")

            while not self._stop_event.is_set():
                loop_start_time = time.time()

                # Get cursor position from breathing pattern
                cursor_pos = self._pattern(loop_start_time)

                # Convert to motor target positions
                ideal_targets_dict, _ = cursor_to_motor_positions(
                    cursor_pos, calibrated_ticks_map
                )

                # Apply motor step limits
                positions_before = dict(self._current_motor_positions)
                final_targets_for_bus = {}
                for motor_id_key in motor_keys:
                    current_pos_m = self._current_motor_positions[motor_id_key]
                    ideal_target_m = ideal_targets_dict[motor_id_key]
                    desired_step = ideal_target_m - current_pos_m

                    clipped_step = np.clip(
                        desired_step,
                        -self.max_motor_step_per_loop,
                        self.max_motor_step_per_loop,
                    )

                    final_target_m = current_pos_m + clipped_step
                    final_targets_for_bus[motor_id_key] = final_target_m
                    self._current_motor_positions[motor_id_key] = final_target_m

                # Send motor commands.
                #
                # Survive a failed frame rather than dying. This loop runs in a
                # thread, so an exception here killed the thread outright: idle
                # motion stopped while the rest of the process carried on, which
                # presents as "the robot went still", not as a crash. A refused
                # out-of-range target, a dropped bus write or a transient serial
                # error should cost one frame, not the whole behaviour.
                #
                # _current_motor_positions is rolled back on failure, because it
                # was already advanced by the step limiter above; leaving it
                # advanced would make the loop believe it had moved and compound
                # the error on the next frame.
                try:
                    self.motor_controller.set_positions(final_targets_for_bus)
                except Exception as exc:
                    self._consecutive_errors = getattr(
                        self, "_consecutive_errors", 0) + 1
                    for k in motor_keys:
                        self._current_motor_positions[k] = positions_before[k]
                    if self._consecutive_errors in (1, 10) or (
                        self._consecutive_errors % 100 == 0
                    ):
                        logger.warning(
                            "Idle motion: command failed (%d consecutive): %s",
                            self._consecutive_errors, exc,
                        )
                    time.sleep(dt_loop)
                    continue
                self._consecutive_errors = 0

                actual_positions = self.motor_controller.get_positions()
                if actual_positions:
                    self._current_motor_positions.update(actual_positions)

                # Sleep to maintain loop frequency
                elapsed = time.time() - loop_start_time
                sleep_time = dt_loop - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            logger.error("Error in idle motion loop: %s", e)
            raise
        finally:
            logger.info("Idle motion loop ended")


@app.command()
def test_idle(
    duration: int = typer.Option(
        10, "--duration", "-d", help="Duration to run idle motion (seconds)"
    ),
    hardware_config: Optional[str] = typer.Option(
        None, "--hardware-config", "-h", help="Path to hardware configuration file"
    ),
    control_config: Optional[str] = typer.Option(
        None, "--control-config", "-c", help="Path to control configuration file"
    ),
) -> None:
    """Test idle motion with breathing pattern."""
    console.print(f"[bold blue]Testing Idle Motion for {duration} seconds[/bold blue]")

    try:
        # Create hardware config
        from ..configs.loaders import get_hardware_config, get_control_config

        hardware_cfg = get_hardware_config(hardware_config)
        control_cfg = get_control_config(control_config)

        console.print(f"Connecting to motors on port: [cyan]{hardware_cfg.port}[/cyan]")

        if control_config:
            console.print(f"Using control config: [cyan]{control_config}[/cyan]")
            console.print(f"Pattern config loaded: [green]✓[/green]")
        else:
            console.print(
                "[yellow]Warning: No control config provided, using defaults[/yellow]"
            )

        # Connect to motors
        with console.status("[bold green]Connecting..."):
            motor_controller = MotorController(hardware_cfg)
            motor_controller.connect()

        console.print("[green]✓[/green] Connected to motors")

        # Create and start idle motion loop with proper breathing pattern config
        console.print("[bold yellow]Starting idle motion...[/bold yellow]")
        idle_loop = IdleMotionLoop(
            motor_controller=motor_controller,
            pattern_config=control_cfg.idle_motion_pattern_config,  # Use control config instead of {}
            hz=hardware_cfg.idle_motion_hz,
            max_motor_step_per_loop=control_cfg.idle_motion_max_motor_step_per_loop,
        )

        idle_loop.start()

        try:
            # Run for specified duration
            time.sleep(duration)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")

        # Stop idle motion
        console.print("[dim]Stopping idle motion...[/dim]")
        idle_loop.stop(reset_to_calibrated=True)

        console.print("[green]✓[/green] Idle motion test completed")

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)
