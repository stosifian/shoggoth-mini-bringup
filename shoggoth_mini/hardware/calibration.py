"""Motor calibration utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, List
from ..common.constants import (
    MOTOR_NAMES,
    MOTOR_ONE_FULL_TURN_TICKS,
    MOTOR_POSITION_MIN,
    MOTOR_POSITION_MAX,
)

logger = logging.getLogger(__name__)


def load_calibration(
    calibration_file_path: Path = Path(
        "assets/hardware/calibration/tentacle_calibration.json"
    ),
    motor_names: List[str] = MOTOR_NAMES,
) -> Dict[str, int]:
    """Load motor calibration data from JSON file.

    Args:
        calibration_file_path: Path to calibration JSON file
        motor_names: List of motor names to load calibration for

    Returns:
        Dictionary mapping motor names to calibrated tick positions

    Raises:
        ValueError: If calibration file is corrupted or invalid
    """
    calibrated_ticks_map = {motor_name: 0 for motor_name in motor_names}

    try:
        if not calibration_file_path.exists():
            logger.info(
                "Calibration file '%s' not found. Motors will start at position 0.",
                calibration_file_path,
            )
            return calibrated_ticks_map

        with open(calibration_file_path, "r") as f:
            raw_calibrated_data = json.load(f)

        for motor_name_key, data_val in raw_calibrated_data.items():
            if motor_name_key in motor_names:
                if isinstance(data_val, dict) and "ticks" in data_val:
                    calibrated_ticks_map[motor_name_key] = int(data_val["ticks"])
                elif isinstance(data_val, (int, float)):
                    # Support simple format where value is just the tick count
                    calibrated_ticks_map[motor_name_key] = int(data_val)
                else:
                    logger.warning(
                        "Invalid calibration data for %s: %s", motor_name_key, data_val
                    )

        logger.info("Loaded calibration data from '%s'", calibration_file_path)
        for motor_name, ticks in calibrated_ticks_map.items():
            logger.debug("Motor %s: %d ticks", motor_name, ticks)

    except json.JSONDecodeError as e:
        logger.error("Error decoding JSON from '%s': %s", calibration_file_path, e)
        raise ValueError(f"Error decoding JSON from '{calibration_file_path}': {e}")
    except Exception as e:
        logger.error(
            "Error loading calibration data from '%s': %s", calibration_file_path, e
        )
        raise ValueError(
            f"Error loading calibration data from '{calibration_file_path}': {e}"
        )

    return calibrated_ticks_map


def save_calibration(
    calibrated_ticks_map: Dict[str, int],
    calibration_file_path: Path = Path(
        "assets/hardware/calibration/tentacle_calibration.json"
    ),
) -> None:
    """Save motor calibration data to JSON file.

    Args:
        calibrated_ticks_map: Dictionary mapping motor names to tick positions
        calibration_file_path: Path where to save calibration data

    Raises:
        ValueError: If unable to save calibration file
    """
    try:
        # Create directory if it doesn't exist
        calibration_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to the expected format with ticks and turns
        calibration_data = {}
        for motor_name, ticks in calibrated_ticks_map.items():
            calibration_data[motor_name] = {
                "ticks": int(ticks),
                "turns": float(ticks) / MOTOR_ONE_FULL_TURN_TICKS,
            }

        with open(calibration_file_path, "w") as f:
            json.dump(calibration_data, f, indent=2)

        logger.info("Saved calibration data to '%s'", calibration_file_path)
        for motor_name, data in calibration_data.items():
            logger.debug(
                "Motor %s: %d ticks (%.6f turns)",
                motor_name,
                data["ticks"],
                data["turns"],
            )

    except Exception as e:
        logger.error(
            "Error saving calibration data to '%s': %s", calibration_file_path, e
        )
        raise ValueError(
            f"Error saving calibration data to '{calibration_file_path}': {e}"
        )


def validate_calibration(calibrated_ticks_map: Dict[str, int]) -> bool:
    """Validate that calibration data is reasonable.

    Args:
        calibrated_ticks_map: Dictionary mapping motor names to tick positions

    Returns:
        True if calibration data appears valid
    """
    for motor_name, ticks in calibrated_ticks_map.items():
        if not isinstance(ticks, int):
            logger.warning(
                "Motor %s calibration is not an integer: %s", motor_name, ticks
            )
            return False
        # Range is the PHYSICAL encoder range, not the raw 16-bit register range.
        # This previously checked MOTOR_POSITION_MIN/MAX (+/-32768), which accepted
        # a saved calibration of -1548 on motor 2 (2026-08-18). Every cursor target
        # derived from it was then a negative absolute position, which geometry.py
        # correctly encodes as sign-magnitude and the servo correctly obeys — so a
        # 0.18-magnitude sweep commanded a 4234-tick move to -2186 with the tendons
        # threaded. Nothing downstream can tell a poisoned zero from a good one.
        if not (0 <= ticks < MOTOR_ONE_FULL_TURN_TICKS):
            logger.warning(
                "Motor %s calibration %d is outside the encoder range 0..%d. "
                "Targets derived from it will be unreachable positions.",
                motor_name,
                ticks,
                MOTOR_ONE_FULL_TURN_TICKS - 1,
            )
            return False

    return True
