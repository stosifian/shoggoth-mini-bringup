"""Motor calibration utility with interactive interface."""

import signal
import sys
import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
import logging

from ..hardware.motors import MotorController
from ..hardware.calibration import save_calibration
from ..configs import get_hardware_config
from ..common.constants import MOTOR_NAMES


logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer(help="Motor calibration utility")

# Calibration constants
BASE_SPEED_STEPS = 800
WHEEL_MODE = 1
SERVO_MODE = 0

KEY_TO_MOTORS = {
    "left": ["1"],
    "up": ["2"],
    "right": ["3"],
}


class TentacleCalibrator:
    """Keyboard-controlled tentacle calibrator."""

    def __init__(self, motor_controller: MotorController, output_file: Path):
        """Initialize calibrator with motor controller."""
        self.motor_controller = motor_controller
        self.output_file = output_file
        self.direction = +1  # +1 = CCW, −1 = CW
        self.pressed_keys = set()
        self.running = True

        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._abort)

        # Initialize motors to wheel mode for calibration
        self._initialize_wheel_mode()

    def _initialize_wheel_mode(self):
        """Set motors to wheel mode for calibration."""
        console.print(
            "[yellow]Setting motors to wheel mode for calibration...[/yellow]"
        )
        for motor_name in MOTOR_NAMES:
            try:
                self.motor_controller._motor_bus.write("Mode", WHEEL_MODE, motor_name)
                self.motor_controller._motor_bus.write("Goal_Speed", 0, motor_name)
                # ONLY EDIT TO THIS FILE. Writing Mode CLEARS Torque_Enable on these
                # servos (firmware interlock, confirmed 2026-08-12), and nothing in
                # the codebase ever sets torque. Without this line the servo accepts
                # Goal_Speed and applies no current: the arrow keys register, the
                # tool prints "Turning 1", and the motors sit silent with no error
                # anywhere. Torque goes on AFTER the Mode write (the interlock would
                # eat it otherwise) and after Goal_Speed is zeroed, so enabling it
                # cannot lurch the tendons.
                self.motor_controller._motor_bus.write("Torque_Enable", 1, motor_name)
            except Exception as e:
                console.print(
                    f"[red]Warning: Could not set {motor_name} to wheel mode: {e}[/red]"
                )

    def _signed_speed(self, direction: int, magnitude: int) -> int:
        """Return a signed int for Goal_Speed, ensuring symmetric magnitude."""
        magnitude = min(abs(magnitude), 1023)
        if direction >= 0:  # CCW
            return magnitude
        # CW
        return -(1024 - magnitude)

    def _update_speeds(self):
        """Update motor speeds based on pressed keys."""
        active_motors = {
            m for key in self.pressed_keys for m in KEY_TO_MOTORS.get(key, [])
        }
        speed_value = self._signed_speed(self.direction, BASE_SPEED_STEPS)

        for motor_name in MOTOR_NAMES:
            val = speed_value if motor_name in active_motors else 0
            if motor_name in active_motors:
                console.print(f"Turning {motor_name}")
            try:
                self.motor_controller._motor_bus.write("Goal_Speed", val, motor_name)
            except Exception as e:
                console.print(f"[red]Error controlling {motor_name}: {e}[/red]")

    def _handle_key_input(self):
        """Handle keyboard input using keyboard library."""
        try:
            from pynput import keyboard
        except ImportError:
            console.print(
                "[red]pynput library required. Install with: pip install pynput[/red]"
            )
            return False

        def on_press(key):
            if key == keyboard.Key.space:
                self.direction *= -1
                direction_str = "CW" if self.direction < 0 else "CCW"
                console.print(f"[blue]Direction → {direction_str}[/blue]")
                if self.pressed_keys:
                    self._update_speeds()
                return

            if key == keyboard.Key.enter:
                self.running = False
                return False  # Stop listener

            # Handle arrow keys
            key_name = None
            if key == keyboard.Key.left:
                key_name = "left"
            elif key == keyboard.Key.up:
                key_name = "up"
            elif key == keyboard.Key.right:
                key_name = "right"

            if key_name:
                self.pressed_keys.add(key_name)
                self._update_speeds()

        def on_release(key):
            key_name = None
            if key == keyboard.Key.left:
                key_name = "left"
            elif key == keyboard.Key.up:
                key_name = "up"
            elif key == keyboard.Key.right:
                key_name = "right"

            if key_name and key_name in self.pressed_keys:
                self.pressed_keys.remove(key_name)
                self._update_speeds()

        console.print("\n[bold green]Keyboard Controls:[/bold green]")
        console.print("← ↑ → : motor 1 / motor 2 / motor 3 (hold combinations)")
        console.print("<space> : toggle direction | <enter> : save & quit")
        console.print(
            "\n[yellow]Starting calibration... Press Enter when done.[/yellow]"
        )

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

        return True

    def _stop_and_save(self):
        """Stop motors and save calibration."""
        console.print("\n[yellow]Stopping motors and saving calibration...[/yellow]")

        # Stop all motors
        for motor_name in MOTOR_NAMES:
            try:
                self.motor_controller._motor_bus.write("Goal_Speed", 0, motor_name)
                self.motor_controller._motor_bus.write("Mode", SERVO_MODE, motor_name)
            except Exception as e:
                console.print(f"[red]Warning: Could not stop {motor_name}: {e}[/red]")

        # Read current positions
        calibration_data = {}
        console.print("\n[bold]Reading current motor positions:[/bold]")

        for motor_name in MOTOR_NAMES:
            try:
                # Try to read angle/position
                ticks = None
                for field in ("Present_Angle", "Present_Position"):
                    try:
                        ticks = int(
                            self.motor_controller._motor_bus.read(field, motor_name)
                        )
                        break
                    except (KeyError, Exception):
                        continue

                if ticks is None:
                    # Fallback to our motor controller method
                    ticks = self.motor_controller.get_position(motor_name)

                calibration_data[motor_name] = ticks
                console.print(f"  {motor_name}: {ticks} ticks")

            except Exception as e:
                console.print(f"[red]Error reading {motor_name} position: {e}[/red]")
                calibration_data[motor_name] = 0

        # Save calibration
        save_calibration(calibration_data, self.output_file)
        console.print(f"\n[green]✓[/green] Calibration saved to {self.output_file}")

    def _abort(self, *_):
        """Handle Ctrl+C abort."""
        console.print("\n[yellow]Aborting calibration...[/yellow]")
        for motor_name in MOTOR_NAMES:
            try:
                self.motor_controller._motor_bus.write("Goal_Speed", 0, motor_name)
                self.motor_controller._motor_bus.write("Mode", SERVO_MODE, motor_name)
            except Exception:
                pass
        console.print("[yellow]Motors stopped.[/yellow]")
        sys.exit(0)

    def run(self):
        """Run the calibration process."""
        if self._handle_key_input():
            self._stop_and_save()


@app.command()
def calibrate(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
) -> None:
    """Calibrate motor zero positions using keyboard controls."""
    console.print("[bold blue]Motor Calibration Tool[/bold blue]")
    console.print("Original keyboard-based calibration restored.")

    # Create hardware config from file
    hardware_config = get_hardware_config(config)

    console.print(f"Using port: [cyan]{hardware_config.port}[/cyan]")
    console.print(f"Output file: [cyan]{hardware_config.calibration_file}[/cyan]")

    try:
        # Connect to motors
        with console.status("[bold green]Connecting to motors..."):
            motor_controller = MotorController(hardware_config)
            motor_controller.connect()

        console.print("[green]✓[/green] Connected to motors successfully")

        # Run calibration
        calibrator = TentacleCalibrator(
            motor_controller, hardware_config.calibration_file
        )
        calibrator.run()

    except KeyboardInterrupt:
        console.print("\n[yellow]Calibration cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
