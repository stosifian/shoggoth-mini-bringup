"""Motor controller abstraction for Feetech servos."""

import logging
import threading
import time
from typing import Dict, Optional, Tuple

from ..configs.hardware import HardwareConfig
from ..common.constants import MOTOR_NAMES, MOTOR_ONE_FULL_TURN_TICKS
from .calibration import load_calibration, validate_calibration

logger = logging.getLogger(__name__)


class MotorController:
    """Unified motor controller for Feetech servos."""

    def __init__(self, config: Optional[HardwareConfig] = None):
        """Initialize motor controller.

        Args:
            config: Hardware configuration, will create default if None
        """
        self.config = config or HardwareConfig()
        self._motor_bus = None
        self._is_connected = False
        self._calibration_data: Dict[str, int] = {}
        # (KEPT CHANGE 1 of 3) One half-duplex serial bus, three concurrent writers:
        # the idle-motion thread, primitives from the orchestrator's async handler,
        # and the closed-loop controller thread. Nothing serialised them, which
        # produced "[TxRxResult] Port is in use!" and killed the idle thread with an
        # unhandled exception (2026-08-12). RLock because set_positions nests into
        # set_position. Does NOT change any commanded value.
        self._bus_lock = threading.RLock()

    def connect(self) -> None:
        """Connect to motor bus and initialize motors.

        Raises:
            RuntimeError: If connection fails or motors not found
        """
        try:
            # Import Feetech dependencies
            bus_config_cls, bus_cls, _, _ = self._get_motor_bus_classes()

            # Create motor bus configuration
            bus_config = bus_config_cls(
                port=self.config.port,
                motors=self.config.motor_config,
            )

            # Create and connect motor bus
            self._motor_bus = bus_cls(bus_config)
            self._motor_bus.connect()
            logger.info("Connected to motor bus on port %s", self.config.port)

            # Set baudrate
            self._motor_bus.set_bus_baudrate(self.config.baudrate)

            # Verify all motors are present
            self._verify_motors()

            # Initialize motors
            self._initialize_motors()

            # Load calibration
            self._load_calibration()

            self._is_connected = True

        except Exception as e:
            logger.error("Failed to connect to motors: %s", e)
            raise RuntimeError(f"Failed to connect to motors: {e}")

    def disconnect(self) -> None:
        """Disconnect from motor bus."""
        if self._motor_bus and self._is_connected:
            try:
                self._motor_bus.disconnect()
                logger.info("Disconnected from motor bus")
            except Exception as e:
                logger.warning("Error during disconnect: %s", e)
            finally:
                self._is_connected = False
                self._motor_bus = None

    def set_position(self, motor_name: str, position: int) -> None:
        """Set target position for a single motor.

        Args:
            motor_name: Name of the motor
            position: Target position in ticks

        Raises:
            RuntimeError: If motor communication fails
        """
        if not self._is_connected:
            raise RuntimeError("Motor controller not connected")

        if motor_name not in self.config.motor_config:
            raise RuntimeError(f"Unknown motor: {motor_name}")

        # (KEPT CHANGE 3 of 3) Refuse targets outside the encoder range.
        #
        # This REFUSES; it never rewrites. A clamp would silently alter the
        # commanded pose, and in a differential three-tendon system clamping one
        # motor while the others move produces a pose nobody asked for.
        #
        # Measured 2026-08-18: grab at the upstream 0.7 commanded motor 2 to 4915,
        # past the fold. Present_Position then reads negative, so the next
        # read-modify-write in any control path computes a negative target — and a
        # raw negative Goal_Position is decoded SIGN-MAGNITUDE by this firmware,
        # turning -3217 into a target of -29551. The motor runs away at full speed
        # until it jams.
        #
        # Negative values are refused too, including geometry.py's
        # `-32768 - absolute` encoding. That encoding is correct for the wire
        # format, but with the calibrated zero at mid-range it should never fire;
        # if it does, the calibration or the cursor magnitude is wrong and the
        # right outcome is a stop, not a move.
        if not 0 <= position < MOTOR_ONE_FULL_TURN_TICKS:
            raise RuntimeError(
                f"Refusing out-of-range Goal_Position {position} for motor "
                f"{motor_name} (valid 0..{MOTOR_ONE_FULL_TURN_TICKS - 1}). "
                f"Past the encoder fold the position reading goes negative and "
                f"every read-modify-write path then commands a runaway."
            )

        try:
            with self._bus_lock:
                self._motor_bus.write("Goal_Position", position, motor_name)
        except Exception as e:
            logger.error("Failed to set position for motor %s: %s", motor_name, e)
            raise RuntimeError(f"Failed to set position for {motor_name}: {e}")

    def set_positions(self, positions: Dict[str, int]) -> None:
        """Set target positions for multiple motors.

        Args:
            positions: Dictionary mapping motor names to target positions

        Raises:
            RuntimeError: If motor communication fails
        """
        with self._bus_lock:
            for motor_name, position in positions.items():
                self.set_position(motor_name, position)

    def get_position(self, motor_name: str) -> int:
        """Get current position of a motor.

        Args:
            motor_name: Name of the motor

        Returns:
            Current position in ticks

        Raises:
            RuntimeError: If motor communication fails
        """
        if not self._is_connected:
            raise RuntimeError("Motor controller not connected")

        if motor_name not in self.config.motor_config:
            raise RuntimeError(f"Unknown motor: {motor_name}")

        try:
            # (KEPT CHANGE 2 of 3) The bus returns a numpy array; this signature
            # promises int. Returning the array crashed %d log formatting, which
            # swallowed the very warning that reported a missed target (2026-08-12).
            # Pure type coercion — the VALUE is unchanged.
            with self._bus_lock:
                raw = self._motor_bus.read("Present_Position", motor_name)
            if hasattr(raw, "item"):
                return int(raw.item()) if raw.size == 1 else int(raw[0])
            return int(raw)
        except Exception as e:
            logger.error("Failed to read position for motor %s: %s", motor_name, e)
            raise RuntimeError(f"Failed to read position for {motor_name}: {e}")

    def get_positions(self) -> Dict[str, int]:
        """Get current positions of all motors.

        Returns:
            Dictionary mapping motor names to current positions
        """
        positions = {}
        with self._bus_lock:
            for motor_name in self.config.motor_config.keys():
                positions[motor_name] = self.get_position(motor_name)
        return positions

    def reset_to_calibrated_positions(self) -> None:
        """Reset all motors to their calibrated start positions."""
        if not self._calibration_data:
            logger.warning("No calibration data loaded, resetting to position 0")
            positions = {name: 0 for name in self.config.motor_config.keys()}
        else:
            positions = self._calibration_data.copy()

        logger.info("Resetting motors to calibrated start positions")
        self.set_positions(positions)

        # Wait for motors to settle
        time.sleep(self.config.motor_settle_time)

        # Verify positions
        self._verify_positions(positions)

    def get_calibration_data(self) -> Dict[str, int]:
        """Get the loaded calibration data.

        Returns:
            Dictionary mapping motor names to calibrated positions
        """
        return self._calibration_data.copy()

    @property
    def is_connected(self) -> bool:
        """Check if motor controller is connected."""
        return self._is_connected

    def _get_motor_bus_classes(self) -> Tuple:
        """Get motor bus classes for Feetech motors."""
        try:
            from lerobot.common.robot_devices.motors.configs import (
                FeetechMotorsBusConfig,
            )
            from lerobot.common.robot_devices.motors.feetech import (
                MODEL_BAUDRATE_TABLE,
                SCS_SERIES_BAUDRATE_TABLE,
                FeetechMotorsBus,
            )

            return (
                FeetechMotorsBusConfig,
                FeetechMotorsBus,
                MODEL_BAUDRATE_TABLE,
                SCS_SERIES_BAUDRATE_TABLE,
            )
        except ImportError as e:
            logger.error("Failed to import Feetech motor dependencies: %s", e)
            raise RuntimeError(f"Failed to import Feetech motor dependencies: {e}")

    def _verify_motors(self) -> None:
        """Verify that all configured motors are present."""
        motor_ids = [config[0] for config in self.config.motor_config.values()]
        present_ids = self._motor_bus.find_motor_indices(motor_ids)

        if len(present_ids) != len(motor_ids):
            missing_ids = set(motor_ids) - set(present_ids)
            error_msg = f"Not all motors found. Expected: {motor_ids}, Present: {present_ids}, Missing: {list(missing_ids)}. Check connections and IDs."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info("All motors found: %s", present_ids)

    def _initialize_motors(self) -> None:
        """Initialize motor settings for full rotation control."""
        logger.info("Setting motors for full rotation control")

        for motor_name in self.config.motor_config.keys():
            try:
                # Set angle limits to 0 to enable full rotation
                self._motor_bus.write("Min_Angle_Limit", 0, motor_name)
                self._motor_bus.write("Max_Angle_Limit", 0, motor_name)
            except Exception as e:
                logger.error("Failed to initialize motor %s: %s", motor_name, e)
                raise RuntimeError(f"Failed to initialize {motor_name}: {e}")

        time.sleep(0.2)  # Give motors time to process the commands
        logger.info("Motors initialized for full rotation control")

    def _load_calibration(self) -> None:
        """Load motor calibration data."""
        try:
            self._calibration_data = load_calibration(
                self.config.calibration_file, MOTOR_NAMES
            )

            # REFUSE, do not warn. A warning here was silently ignored while a
            # calibration of -1548 on motor 2 drove a 4234-tick move with the
            # tendons threaded (2026-08-18). This is a pre-flight check: it stops
            # a run before anything moves and never alters a command in flight.
            if not validate_calibration(self._calibration_data):
                raise RuntimeError(
                    f"Refusing to connect: calibration in "
                    f"{self.config.calibration_file} is invalid "
                    f"({self._calibration_data}). Every cursor target is computed "
                    f"as an offset from these values, so a bad zero commands the "
                    f"motors to unreachable positions. Re-run calibration."
                )

        except RuntimeError:
            raise
        except Exception as e:
            logger.warning("Failed to load calibration: %s", e)
            # Use zero positions as fallback
            self._calibration_data = {
                name: 0 for name in self.config.motor_config.keys()
            }

    def _verify_positions(self, target_positions: Dict[str, int]) -> None:
        """Verify that motors reached their target positions."""
        try:
            for motor_name, target_pos in target_positions.items():
                actual_pos = self.get_position(motor_name)
                if abs(actual_pos - target_pos) > self.config.position_tolerance:
                    logger.warning(
                        "Motor %s did not reach target %d (currently at %d)",
                        motor_name,
                        target_pos,
                        actual_pos,
                    )
                    # Try setting position again
                    self.set_position(motor_name, target_pos)
                else:
                    logger.debug(
                        "Motor %s reached target %d (actual: %d)",
                        motor_name,
                        target_pos,
                        actual_pos,
                    )
        except Exception as e:
            logger.warning("Could not verify motor positions: %s", e)

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
