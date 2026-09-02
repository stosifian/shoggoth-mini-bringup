"""Closed-loop control system combining vision and motor control."""

import csv
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import typer
from rich.console import Console
from stable_baselines3 import PPO

from ..perception.detection import YOLODetector
from ..perception.hand_tracking import get_mediapipe_hand_data, close_mediapipe_hands
from ..perception.stereo import (
    triangulate_points,
    load_stereo_calibration,
    split_stereo_frame,
)

from ..hardware.motors import MotorController
from ..common.constants import MOTOR_NAMES
from ..perception.camera import read_oriented
from ..perception.velocity_gate import VelocityGate

from .geometry import cursor_to_motor_positions

from shoggoth_mini.configs import (
    get_control_config,
    get_perception_config,
    get_hardware_config,
)

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer(help="Closed-loop RL control")


@dataclass
class CableMapper:
    """Utility for converting motor ticks to cable lengths for observations."""

    ticks_per_rotation: int
    length_per_rotation: float
    baseline_length: float
    tick_sign: int = 1

    length_per_tick: float = field(init=False)

    def __post_init__(self):
        self.length_per_tick = self.length_per_rotation / float(self.ticks_per_rotation)

    def ticks_to_length(self, ticks: int, calibrated_zero_ticks: int) -> float:
        """Converts *absolute* motor ticks to cable length in metres."""
        rel_ticks = ticks - calibrated_zero_ticks
        return self.baseline_length + rel_ticks * self.length_per_tick * self.tick_sign

    def get_cable_lengths_from_positions(
        self,
        positions_ticks: Dict[str, int],
        calib_ticks_map: Dict[str, int],
    ) -> np.ndarray:
        """Convert motor positions to cable lengths array for RL observations."""
        lengths = np.zeros(3, dtype=np.float32)
        for motor_name in MOTOR_NAMES:
            if motor_name in positions_ticks:
                sim_idx = int(motor_name) - 1
                lengths[sim_idx] = self.ticks_to_length(
                    positions_ticks[motor_name], calib_ticks_map[motor_name]
                )
        return lengths


class ObservationBuffer:
    """Maintains a fixed-length buffer of stacked observation frames."""

    def __init__(self, num_frames: int, frame_shape: int):
        self.buffer: Deque[np.ndarray] = deque(maxlen=num_frames)
        self.num_frames = num_frames
        self.frame_shape = frame_shape

    def _build_single_frame_obs(
        self,
        tip_pos_m: np.ndarray,
        target_pos_m: np.ndarray,
        actuator_lengths_m: Optional[np.ndarray] = None,
        clip_obs: bool = False,
        clip_bounds: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Formats a single observation frame, optionally clipping values."""

        # Transform tip and target positions to match the policy's input format
        tip_pos_transformed = np.array(
            [tip_pos_m[2], tip_pos_m[0], tip_pos_m[1]], dtype=np.float32
        )
        target_pos_transformed = np.array(
            [target_pos_m[2], target_pos_m[0], target_pos_m[1]], dtype=np.float32
        )

        obs_parts = [tip_pos_transformed, target_pos_transformed]

        obs_parts.append(actuator_lengths_m.astype(np.float32).copy())

        raw_obs_frame = np.concatenate(obs_parts).astype(np.float32)
        raw_obs_frame = np.round(raw_obs_frame, 2)

        if clip_obs and clip_bounds is not None:
            raw_obs_frame[0:3] = np.clip(
                raw_obs_frame[0:3],
                clip_bounds.get("tip_min", -np.inf),
                clip_bounds.get("tip_max", np.inf),
            )
            raw_obs_frame[3:6] = np.clip(
                raw_obs_frame[3:6],
                clip_bounds.get("target_min", -np.inf),
                clip_bounds.get("target_max", np.inf),
            )
            if actuator_lengths_m is not None and len(raw_obs_frame) >= 9:
                raw_obs_frame[6:9] = np.clip(
                    raw_obs_frame[6:9],
                    clip_bounds.get("actuator_length_min", -np.inf),
                    clip_bounds.get("actuator_length_max", np.inf),
                )

        return raw_obs_frame

    def append(
        self,
        tip_pos_m: np.ndarray,
        target_pos_m: np.ndarray,
        actuator_lengths_m: Optional[np.ndarray] = None,
        clip_obs: bool = False,
        clip_bounds: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Build observation frame and add to buffer, returning stacked observation."""
        frame = self._build_single_frame_obs(
            tip_pos_m=tip_pos_m,
            target_pos_m=target_pos_m,
            actuator_lengths_m=actuator_lengths_m,
            clip_obs=clip_obs,
            clip_bounds=clip_bounds,
        )

        self.buffer.append(frame.astype(np.float32).copy())

        # For the first few frames, we need to pad the buffer
        while len(self.buffer) < self.num_frames:
            self.buffer.append(frame.astype(np.float32).copy())

        return np.concatenate(list(self.buffer), axis=0)


class PerceptionSystem:
    """Handles camera input, object detection, and stereo vision processing."""

    def __init__(
        self,
        perception_config,
        camera_index: Optional[int] = None,
        external_camera: Optional[cv2.VideoCapture] = None,
    ):
        """Initialize perception system.

        Args:
            perception_config: Perception configuration object
            camera_index: Camera index to use (overrides config if provided)
            external_camera: External camera object to use instead of creating new one
        """
        self.config = perception_config
        # Per-eye detection status and the frame it came from. triangulate_points()
        # returns None when EITHER eye is missing, so a single "finger lost" flag
        # hides the difference between "detector found nothing" and "found it in one
        # eye only" (occlusion / outside the stereo overlap) — different causes.
        self.last_flags = {}
        self.last_frame = None
        self.cap = external_camera
        self.owns_camera = external_camera is None
        self.executor = None

        # Initialize camera if not provided externally
        if self.cap is None:
            camera_idx = (
                camera_index if camera_index is not None else self.config.camera_index
            )
            self.cap = cv2.VideoCapture(camera_idx)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {camera_idx}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.stereo_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.stereo_resolution[1])

        # Get frame dimensions
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.half_width = frame_width // 2

        # Initialize perception modules
        self.stereo_calib = load_stereo_calibration(
            calib_dir=self.config.camera_calibration_path
        )
        # One gate per tracked point. Applied AFTER triangulation, so the limit is
        # expressed in real metres per second rather than pixels.
        gate_on = getattr(perception_config, "velocity_gate_enabled", False)
        self.tip_gate = (
            VelocityGate(perception_config.tip_max_speed_m_s,
                         perception_config.velocity_gate_max_rejects, name="tip")
            if gate_on else None
        )
        self.target_gate = (
            VelocityGate(perception_config.target_max_speed_m_s,
                         perception_config.velocity_gate_max_rejects, name="target")
            if gate_on else None
        )

        self.detector = YOLODetector(
            model_path=str(self.config.yolo_model_path),
            device=self.config.yolo_device,
            confidence_threshold=self.config.confidence_threshold,
        )

    def start_parallel_processing(self):
        """Start thread pool for parallel processing."""
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=4)

    def get_detections(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get tip and finger positions from current camera frame.

        Returns:
            Tuple of (tip_position, finger_position) in world coordinates
        """
        if not self.cap:
            return None, None

        ok, frame = read_oriented(self.cap)
        if not ok:
            return None, None

        left_frame, right_frame = split_stereo_frame(frame)

        if self.executor:
            # Parallel processing
            future_left_tip = self.executor.submit(self.detector.detect, left_frame)
            future_right_tip = self.executor.submit(self.detector.detect, right_frame)
            future_left_finger = self.executor.submit(
                get_mediapipe_hand_data, left_frame
            )
            future_right_finger = self.executor.submit(
                get_mediapipe_hand_data, right_frame
            )

            xy_l_tip, _, _ = future_left_tip.result()
            xy_r_tip, _, _ = future_right_tip.result()
            xy_l_finger, _ = future_left_finger.result()
            xy_r_finger, _ = future_right_finger.result()
        else:
            # Sequential processing
            xy_l_tip, _, _ = self.detector.detect(left_frame)
            xy_r_tip, _, _ = self.detector.detect(right_frame)
            xy_l_finger, _ = get_mediapipe_hand_data(left_frame)
            xy_r_finger, _ = get_mediapipe_hand_data(right_frame)

        self.last_frame = frame
        self.last_flags = {
            "tip_l": xy_l_tip is not None,
            "tip_r": xy_r_tip is not None,
            "fing_l": xy_l_finger is not None,
            "fing_r": xy_r_finger is not None,
        }

        # Triangulate 3D positions
        tip_pos = triangulate_points(
            xy_l_tip,
            xy_r_tip,
            self.stereo_calib,
            units_to_m=self.config.units_to_meters,
            rotation_angle_deg=self.config.rotation_angle_deg,
            y_translation_m=self.config.y_translation_m,
            coordinate_limits=self.config.coordinate_limits,
        )

        finger_pos = triangulate_points(
            xy_l_finger,
            xy_r_finger,
            self.stereo_calib,
            units_to_m=self.config.units_to_meters,
            rotation_angle_deg=self.config.rotation_angle_deg,
            y_translation_m=self.config.y_translation_m,
            coordinate_limits=self.config.coordinate_limits,
        )

        if self.tip_gate is not None:
            tip_pos = self.tip_gate.update(tip_pos)
            self.last_flags["tip_gated"] = self.tip_gate.last_was_rejected
        if self.target_gate is not None:
            finger_pos = self.target_gate.update(finger_pos)
            self.last_flags["target_gated"] = self.target_gate.last_was_rejected

        return tip_pos, finger_pos

    def gate_stats(self):
        """Gate counters for the end-of-run summary, or None when disabled."""
        return [g.stats for g in (self.tip_gate, self.target_gate) if g is not None]

    def cleanup(self):
        """Clean up resources."""
        if self.executor:
            self.executor.shutdown()
            self.executor = None
        if self.cap and self.owns_camera:
            self.cap.release()
            self.cap = None
        close_mediapipe_hands()


class RLInferenceEngine:
    """Handles RL model loading, observation processing, and action inference."""

    def __init__(self, control_config):
        """Initialize RL inference engine."""
        self.config = control_config
        self.policy = None
        self.obs_buffer = None

        # Load RL model
        model_path = self.config.model_path
        if not Path(model_path).exists():
            raise FileNotFoundError(f"SB3 model not found at {model_path}")
        self.policy = PPO.load(model_path)

        # Set up observation buffer
        obs_dim_single = 6
        if self.config.include_actuator_lengths_in_obs:
            obs_dim_single += 3

        self.obs_buffer = ObservationBuffer(self.config.num_frames, obs_dim_single)

        # Prepare observation clipping
        self.clip_observations = self.config.clip_observations
        self.obs_clip_bounds = self.config.obs_clip_bounds.copy()
        if self.clip_observations:
            for key in ["tip_min", "tip_max", "target_min", "target_max"]:
                if key in self.obs_clip_bounds:
                    self.obs_clip_bounds[key] = np.array(
                        self.obs_clip_bounds[key], dtype=np.float32
                    )

        # Action smoothing
        self.previous_action: Optional[np.ndarray] = None
        self.action_smoothing_alpha = control_config.action_smoothing_alpha

        # Action calibration offset
        self.action_2d_offset = np.array(
            control_config.action_2d_offset, dtype=np.float32
        )

    def predict_action(
        self,
        tip_pos: np.ndarray,
        target_pos: np.ndarray,
        actuator_lengths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict action from current state.

        Args:
            tip_pos: Current tip position
            target_pos: Target position
            actuator_lengths: Current actuator lengths (optional)

        Returns:
            Predicted 2D cursor action (smoothed)
        """
        # Build observation
        obs_args = {
            "tip_pos_m": tip_pos.astype(np.float32),
            "target_pos_m": target_pos.astype(np.float32),
            "clip_obs": self.clip_observations,
            "clip_bounds": self.obs_clip_bounds,
        }
        if self.config.include_actuator_lengths_in_obs and actuator_lengths is not None:
            obs_args["actuator_lengths_m"] = actuator_lengths

        obs = self.obs_buffer.append(**obs_args)

        # Get raw policy action
        raw_action, _ = self.policy.predict(
            obs, deterministic=self.config.deterministic
        )
        raw_action = np.asarray(raw_action, dtype=np.float32)

        # Apply action smoothing using exponential moving average
        if self.previous_action is None:
            # First action - no smoothing needed
            smoothed_action = raw_action.copy()
        else:
            # Smooth with previous action: new_action = alpha * raw + (1-alpha) * previous
            smoothed_action = (
                self.action_smoothing_alpha * raw_action
                + (1.0 - self.action_smoothing_alpha) * self.previous_action
            )

        # Update previous action for next iteration
        self.previous_action = smoothed_action.copy()

        # Apply calibration offset
        final_action = smoothed_action + self.action_2d_offset

        return final_action


class MotorSystem:
    """Handles motor control, cable mapping, and safety constraints."""

    def __init__(
        self,
        hardware_config,
        control_config,
        external_motor_controller: Optional[MotorController] = None,
    ):
        """Initialize motor system.

        Args:
            hardware_config: Hardware configuration
            control_config: Control configuration
            external_motor_controller: External motor controller to use
        """
        self.hardware_config = hardware_config
        self.control_config = control_config
        self.motor_controller = external_motor_controller
        self.owns_motor_controller = external_motor_controller is None

        if self.motor_controller is None:
            self.motor_controller = MotorController(hardware_config)

        self.cable_mapper = CableMapper(
            ticks_per_rotation=hardware_config.ticks_per_rotation,
            length_per_rotation=hardware_config.length_per_rotation,
            baseline_length=hardware_config.baseline_length,
            tick_sign=hardware_config.tick_sign,
        )

        self.calib_ticks_map = {}
        self.current_positions_ticks = {}
        self.current_cable_lengths_sim_m = None

    def connect(self):
        """Connect to motors and initialize."""
        if self.owns_motor_controller:
            self.motor_controller.connect()
        self.calib_ticks_map = self.motor_controller.get_calibration_data()
        self.current_positions_ticks = self.motor_controller.get_positions()

        # Initialize current cable lengths
        self.current_cable_lengths_sim_m = (
            self.cable_mapper.get_cable_lengths_from_positions(
                self.current_positions_ticks, self.calib_ticks_map
            )
        )

    def safe_set_motor_positions_ticks(self, target_map_ticks: Dict[str, int]):
        """Set motor positions with safety limits."""
        safety_min_off = self.control_config.safety_offset_min
        safety_max_off = self.control_config.safety_offset_max

        guarded_ticks: Dict[str, int] = {}
        for m_name, ticks in target_map_ticks.items():
            zero = self.calib_ticks_map[m_name]
            min_tick = zero + safety_min_off
            max_tick = zero + safety_max_off
            clamped_ticks = np.clip(ticks, min_tick, max_tick)
            if ticks != clamped_ticks:
                console.print(
                    f"Guardrail clamp {m_name}: {ticks} -> {clamped_ticks} "
                    f"(limits [{min_tick}, {max_tick}])"
                )
            guarded_ticks[m_name] = clamped_ticks

        self.motor_controller.set_positions(guarded_ticks)

        # Read back actual motor positions
        actual_positions = self.motor_controller.get_positions()
        self.current_positions_ticks.update(actual_positions)

        # Update cable lengths based on actual positions
        self.current_cable_lengths_sim_m = (
            self.cable_mapper.get_cable_lengths_from_positions(
                self.current_positions_ticks, self.calib_ticks_map
            )
        )

    def execute_action(self, action_2d: np.ndarray) -> np.ndarray:
        """Execute 2D cursor action using direct motor position conversion.

        Args:
            action_2d: 2D cursor action

        Returns:
            New cable lengths after action execution
        """
        # Limit the cursor MAGNITUDE, not the resulting ticks.
        #
        # max_2d_action_magnitude existed only in the RL training environment and
        # in convert_2d_cursor_to_target_lengths; nothing bounded the policy's
        # output on the live path. Trained with an action space of +/-1.0 per
        # component, the policy can emit a magnitude near 1.41, which converts to
        # targets far outside 0..4095 — set_position then refuses, the exception
        # propagates out of the control loop, and tracking stops.
        #
        # Clamping here rather than in safe_set_motor_positions_ticks is
        # deliberate: scaling the cursor scales all three motors together, so the
        # commanded DIRECTION is preserved exactly and only the reach shortens.
        # Clamping per-motor ticks instead moves one tendon while the others
        # follow the original command, producing a pose nobody asked for.
        limit = float(getattr(self.control_config, "max_2d_action_magnitude", 0.0))
        action_2d = np.asarray(action_2d, dtype=float)
        if limit > 0:
            magnitude = float(np.linalg.norm(action_2d))
            if magnitude > limit:
                action_2d = action_2d * (limit / magnitude)

        # Convert 2D action directly to motor positions
        target_positions, _ = cursor_to_motor_positions(
            action_2d,
            self.calib_ticks_map,
            noise_scale=0.0,
        )

        # Send motor commands
        self.safe_set_motor_positions_ticks(target_positions)

        return self.current_cable_lengths_sim_m

    def run_calibration(self) -> Dict[str, int]:
        """Run direct 2D position calibration."""
        return self._calibrate_to_2d_position()

    def _calibrate_to_2d_position(self) -> Dict[str, int]:
        """Move smoothly to configured 2D calibration position with interpolation and noise."""
        calibration_2d_pos = np.array(
            self.control_config.calibration_2d_position, dtype=np.float32
        )

        logger.info(f"Moving to calibration position: {calibration_2d_pos}")

        # Get current 2D position by inferring from current motor positions
        # We'll start from neutral [0, 0] as a reasonable approximation
        start_2d_pos = np.array([0.0, 0.0], dtype=np.float32)

        # Interpolation parameters
        num_steps = 20
        base_noise_scale = 0.04
        step_duration = 0.01

        logger.info(
            f"Interpolating from {start_2d_pos} to {calibration_2d_pos} in {num_steps} steps"
        )

        # Generate interpolated positions with noise for realism
        for i in range(num_steps + 1):
            # Linear interpolation factor
            alpha = i / num_steps

            # Interpolated position
            interp_pos = start_2d_pos + alpha * (calibration_2d_pos - start_2d_pos)

            # Add decreasing noise (more noise at start, less at end for smooth settling)
            noise_scale = base_noise_scale * (1.0 - alpha)

            # Convert to motor positions with noise
            target_positions, _ = cursor_to_motor_positions(
                interp_pos,
                self.calib_ticks_map,
                noise_scale=noise_scale,
            )

            # Apply the interpolated position
            self.safe_set_motor_positions_ticks(target_positions)

            # Small delay between steps for smooth movement
            if i < num_steps:  # Don't sleep after final step
                time.sleep(step_duration)

        logger.info(
            f"Calibration complete. Motors moved to ticks: {self.current_positions_ticks}"
        )
        return self.current_positions_ticks

    def cleanup(self, reset_to_calibrated: bool = True):
        """Clean up motor system.

        Args:
            reset_to_calibrated: Whether to reset motors to calibrated positions
                before disconnecting
        """
        if self.motor_controller and self.owns_motor_controller:
            try:
                # Only reset if still connected and requested
                if self.motor_controller.is_connected and reset_to_calibrated:
                    self.motor_controller.reset_to_calibrated_positions()
            finally:
                if self.motor_controller.is_connected:
                    self.motor_controller.disconnect()


class ClosedLoopController:
    """Main controller that coordinates perception, RL inference, and motor control."""

    def __init__(
        self,
        control_config=None,
        perception_config=None,
        hardware_config=None,
        external_motor_controller: Optional[MotorController] = None,
        external_camera: Optional[cv2.VideoCapture] = None,
    ):
        """Initialize closed loop controller.

        Args:
            control_config: Control configuration
            perception_config: Perception configuration
            hardware_config: Hardware configuration
            external_motor_controller: External motor controller to use
            external_camera: External camera to use
        """
        self.control_config = control_config
        self.perception_config = perception_config
        self.hardware_config = hardware_config

        # Initialize subsystems
        self.perception_system = None
        self.rl_engine = None
        self.motor_system = None

        # External resources
        self.external_motor_controller = external_motor_controller
        self.external_camera = external_camera

        # Control state
        self.running = False
        self.telemetry = None
        self.dump_dir = None
        self._dump_count = 0
        self._last_dump_t = 0.0
        self.home_positions_ticks = None
        self.lost_counter = 0

    def initialize(self):
        """Initialize all subsystems."""
        console.print("[yellow]Initializing closed-loop controller...[/yellow]")

        # Initialize perception system
        self.perception_system = PerceptionSystem(
            self.perception_config, external_camera=self.external_camera
        )
        self.perception_system.start_parallel_processing()

        # Initialize RL engine
        self.rl_engine = RLInferenceEngine(self.control_config)

        # Initialize motor system
        self.motor_system = MotorSystem(
            self.hardware_config,
            self.control_config,
            external_motor_controller=self.external_motor_controller,
        )
        self.motor_system.connect()

        console.print("[green]✓[/green] All subsystems initialized")

    def run_calibration_phase(self):
        """Run calibration to move to configured 2D position."""
        console.print("[yellow]Running calibration phase...[/yellow]")
        self.motor_system.current_positions_ticks = self.motor_system.run_calibration()
        console.print("[green]✓[/green] Calibration complete")

    def _maybe_dump_miss(self):
        """Save the frame behind a detection failure, so you can see WHY.

        A boolean "not detected" tells you nothing about the cause. The image
        shows whether the hand was occluded by the tentacle, out of one eye's
        view, motion-blurred, or simply not where you thought it was.

        Rate-limited to at most one per second: at 14 Hz a bad patch would
        otherwise write hundreds of near-identical 3840x1080 frames.
        """
        if not self.dump_dir:
            return
        now = time.time()
        if now - self._last_dump_t < 1.0:
            return
        frame = getattr(self.perception_system, "last_frame", None)
        if frame is None:
            return

        flags = self.perception_system.last_flags or {}
        # Encode which eyes saw what into the filename — makes the directory
        # listing itself the summary.
        tag = "".join(
            k.replace("fing", "f").replace("tip", "t").replace("_", "")
            for k, v in flags.items() if v
        ) or "none"
        path = self.dump_dir / f"miss_{self._dump_count:04d}_{tag}.jpg"
        try:
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self._dump_count += 1
            self._last_dump_t = now
        except Exception as e:
            logger.warning("Could not write miss frame: %s", e)

    def run_control_loop(self):
        """Run the main control loop."""
        step_time = 1.0 / self.hardware_config.control_loop_hz
        lost_frames_threshold = self.control_config.lost_frames_threshold

        console.print(
            "[bold green]Starting control loop. Press Ctrl+C to stop.[/bold green]"
        )

        while self.running:
            t_loop_start = time.time()

            # Get detections from perception system
            tip_pos, finger_pos = self.perception_system.get_detections()

            if tip_pos is None or finger_pos is None:
                self.lost_counter += 1
                if self.telemetry:
                    # Log WHICH of the two was missing — "lost the tip" and "lost
                    # the finger" have completely different causes.
                    self.telemetry.log(False, tip_pos, finger_pos, None, None,
                                       self.lost_counter,
                                       self.perception_system.last_flags)
                    self._maybe_dump_miss()
                if (
                    self.home_positions_ticks
                    and self.lost_counter >= lost_frames_threshold
                ):
                    if self.control_config.stop_on_prolonged_loss:
                        # Stop controller entirely (for orchestrator usage)
                        console.print(
                            "[yellow]Stopping finger-follow due to prolonged loss "
                            "of finger.[/yellow]"
                        )
                        self.running = False
                        break
                    else:
                        # Return to home position (for standalone usage)
                        console.print("\nTip lost – returning to home positions…")
                        self.motor_system.safe_set_motor_positions_ticks(
                            self.home_positions_ticks
                        )
                        time.sleep(0.3)
                        self.lost_counter = 0
                continue

            # Reset lost counter when both detections are successful
            self.lost_counter = 0
            tip_m = tip_pos.astype(np.float32)
            target_m = finger_pos.astype(np.float32)

            if self.home_positions_ticks is None:
                self.home_positions_ticks = (
                    self.motor_system.current_positions_ticks.copy()
                )

            # Get RL action
            actuator_lengths = None
            if self.control_config.include_actuator_lengths_in_obs:
                actuator_lengths = self.motor_system.current_cable_lengths_sim_m

            action_2d_cursor = self.rl_engine.predict_action(
                tip_m, target_m, actuator_lengths
            )

            # Execute action
            self.motor_system.execute_action(action_2d_cursor)

            if self.telemetry:
                self.telemetry.log(
                    True, tip_m, target_m, action_2d_cursor,
                    self.motor_system.current_positions_ticks, 0,
                    self.perception_system.last_flags,
                )

            # Sleep to maintain loop timing
            elapsed = time.time() - t_loop_start
            sleep_dur = max(0.0, step_time - elapsed)
            time.sleep(sleep_dur)

    def start(self):
        """Start the closed loop controller."""
        if self.running:
            console.print("[yellow]Controller already running[/yellow]")
            return

        try:
            self.running = True
            self.initialize()
            self.run_calibration_phase()
            self.run_control_loop()
        except KeyboardInterrupt:
            console.print("\n[yellow]Keyboard interrupt received. Stopping…[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            raise
        finally:
            self.stop()

    def stop(self, reset_to_calibrated: bool = True):
        """Stop the closed loop controller.

        Args:
            reset_to_calibrated: Whether to reset motors to calibrated positions
        """
        self.running = False
        console.print("[yellow]Stopping closed loop controller...[/yellow]")

        if self.motor_system:
            self.motor_system.cleanup(reset_to_calibrated=reset_to_calibrated)
        if self.perception_system:
            self.perception_system.cleanup()

            cv2.destroyAllWindows()
            console.print("[green]✓[/green] Shutdown complete.")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class LoopTelemetry:
    """Per-iteration CSV log of the whole closed-loop chain.

    Exists because you cannot run debug-perception alongside the controller — the
    camera is exclusive to one process — so the only way to see what the loop is
    reacting to is to log it from inside. Captures the full chain per iteration:
    what was detected, what the policy asked for, and what the motors were told.

    Diagnosing jerk from this: if tip/target jump frame to frame, the input is
    noisy (detector). If they are smooth but action_x/action_y oscillate, the
    policy is reacting badly (try action_smoothing_alpha). If both are smooth but
    the ticks jump, it is the geometry conversion or a rate limit.
    """

    FIELDS = [
        "t", "dt", "detected",
        "tip_x", "tip_y", "tip_z",
        "target_x", "target_y", "target_z",
        "action_x", "action_y",
        "ticks_1", "ticks_2", "ticks_3",
        "lost_counter",
        "tip_l", "tip_r", "fing_l", "fing_r",
        "tip_gated", "target_gated",
    ]

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(self.FIELDS)
        self._t0 = time.time()
        self._t_prev = self._t0
        self.rows = 0

    @staticmethod
    def _vec3(v):
        return ["", "", ""] if v is None else [f"{float(x):.5f}" for x in v[:3]]

    def log(self, detected, tip, target, action, ticks, lost_counter, flags=None):
        now = time.time()
        row = [f"{now - self._t0:.4f}", f"{now - self._t_prev:.4f}", int(bool(detected))]
        row += self._vec3(tip)
        row += self._vec3(target)
        row += ["", ""] if action is None else [f"{float(action[0]):.5f}",
                                                f"{float(action[1]):.5f}"]
        row += ["", "", ""] if ticks is None else [str(int(ticks.get(m, 0)))
                                                   for m in ("1", "2", "3")]
        row.append(lost_counter)
        flags = flags or {}
        row += [int(bool(flags.get(k))) if k in flags else ""
                for k in ("tip_l", "tip_r", "fing_l", "fing_r",
                          "tip_gated", "target_gated")]
        self._w.writerow(row)
        self._t_prev = now
        self.rows += 1
        # Flush aggressively: tools/plot_loop_live.py tails this file while the
        # loop runs, and buffering 50 rows at ~16 Hz would put it 3 s behind.
        # A few hundred bytes per iteration is not a measurable cost.
        if self.rows % 3 == 0:
            self._fh.flush()

    def close(self):
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass


@app.command()
def run(
    control_config: Optional[str] = typer.Option(
        None, "--control-config", "-c", help="Path to control configuration file"
    ),
    perception_config_path: Optional[str] = typer.Option(
        None, "--perception-config", "-p", help="Path to perception configuration file"
    ),
    hardware_config_path: Optional[str] = typer.Option(
        None, "--hardware-config", "-h", help="Path to hardware configuration file"
    ),
    dump_misses: Optional[str] = typer.Option(
        None, "--dump-misses",
        help="Directory to save stereo frames when a detection fails (max 1/sec). "
             "Filenames encode which eyes saw what, e.g. miss_0003_tl_tr_fl.jpg "
             "means the finger was found in the LEFT eye only.",
    ),
    log_csv: Optional[str] = typer.Option(
        None, "--log-csv",
        help="Write per-iteration telemetry (detections, policy action, motor ticks) "
             "to this CSV. debug-perception cannot run alongside the controller "
             "(the camera is exclusive), so this is the way to see what the loop saw.",
    ),
) -> None:
    """Run closed-loop RL control with tentacle robot."""
    console.print("[bold blue]Starting Closed-Loop RL Control[/bold blue]")

    # Load configurations
    config = get_control_config(control_config)
    perception_config = get_perception_config(perception_config_path)
    hardware_config = get_hardware_config(hardware_config_path)

    console.print(f"Control config: [cyan]{control_config}[/cyan]")
    console.print(f"Perception config: [cyan]{perception_config_path}[/cyan]")
    console.print(f"Hardware config: [cyan]{hardware_config_path}[/cyan]")

    # Create and run controller
    with ClosedLoopController(
        control_config=config,
        perception_config=perception_config,
        hardware_config=hardware_config,
    ) as controller:
        if log_csv:
            controller.telemetry = LoopTelemetry(log_csv)
            console.print(f"Telemetry: [cyan]{log_csv}[/cyan]")
        if dump_misses:
            controller.dump_dir = Path(dump_misses)
            controller.dump_dir.mkdir(parents=True, exist_ok=True)
            console.print(f"Miss frames: [cyan]{dump_misses}[/cyan]")
        try:
            controller.start()
        finally:
            try:
                stats = controller.perception_system.gate_stats()
                for st in stats or []:
                    console.print(
                        f"velocity gate [{st['name']}]: accepted {st['accepted']}, "
                        f"rejected {st['rejected']} ({100*st['reject_rate']:.1f}%), "
                        f"force-accepted {st['forced']}"
                    )
            except Exception:
                pass
            if controller.telemetry:
                controller.telemetry.close()
                console.print(
                    f"[green]wrote {controller.telemetry.rows} rows to "
                    f"{log_csv}[/green]"
                )


def main() -> None:
    """Main entry point for the closed-loop control CLI."""
    app()


if __name__ == "__main__":
    main()
