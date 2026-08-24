"""Real-time perception debugging and visualization."""

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import typer
from rich.console import Console

from ..configs.loaders import get_perception_config
from .detection import YOLODetector
from .hand_tracking import get_mediapipe_hand_data, close_mediapipe_hands
from .stereo import load_stereo_calibration, triangulate_points, split_stereo_frame
from .camera import read_oriented
from .dashboard import (
    setup_dashboard_figure,
    draw_detection_overlays,
    assemble_dashboard_view,
)

console = Console()
app = typer.Typer(help="Perception debugging utilities")


def debug_perception_impl(
    camera_index: int,
    model_path: Optional[Path],
    max_trajectory_points: int,
    enable_multithreading: bool,
    config: Optional[str],
):
    """Run real-time perception debugging with stereo triangulation."""

    # Load configuration
    config = get_perception_config(config)

    # Override model path if provided
    if model_path:
        config.yolo_model_path = model_path

    # Validate model path
    if not config.yolo_model_path.exists():
        console.print(f"Error: YOLO model not found at {config.yolo_model_path}")
        console.print(
            "Please check the model path in perception config or provide --model"
        )
        raise typer.Exit(1)

    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        console.print(f"Error: Cannot open camera {camera_index}")
        raise typer.Exit(1)

    # Set camera resolution
    width, height = config.stereo_resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    half_width = actual_width // 2

    console.print(
        f"Camera resolution: {actual_width}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    )
    console.print(
        f"Left/right resolution: {half_width}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    )

    # Load stereo calibration
    try:
        stereo_calib = load_stereo_calibration(calib_dir=config.camera_calibration_path)
        console.print("Stereo calibration loaded successfully")
    except Exception as e:
        console.print(f"Error loading stereo calibration: {e}")
        raise typer.Exit(1)

    # Initialize YOLO detector
    try:
        detector = YOLODetector(
            model_path=str(config.yolo_model_path),
            device=config.yolo_device,
            confidence_threshold=config.confidence_threshold,
        )
        console.print("YOLO detector initialized successfully")
    except Exception as e:
        console.print(f"Error initializing YOLO detector: {e}")
        raise typer.Exit(1)

    # Setup dashboard
    plot_handles = setup_dashboard_figure(
        figure_width=config.dashboard_figure_size[0],
        figure_height=config.dashboard_figure_size[1],
        coordinate_limits=config.coordinate_limits,
    )

    # Initialize trajectory storage
    tip_trajectory = deque(maxlen=max_trajectory_points)
    finger_trajectory = deque(maxlen=max_trajectory_points)

    # Threading setup
    executor = ThreadPoolExecutor(max_workers=4) if enable_multithreading else None

    frame_count = 0
    last_print_time = time.time()

    console.print("Starting perception debugging...")
    console.print("Press 'q' to quit")

    try:
        while True:
            loop_start = time.time()

            # Capture frame
            ret, frame = read_oriented(cap)
            if not ret:
                console.print("Failed to capture frame")
                break

            # Split stereo frame
            left_frame, right_frame = split_stereo_frame(frame)

            detection_start = time.time()

            # Run detections (potentially in parallel)
            if enable_multithreading and executor:
                future_left_tip = executor.submit(detector.detect, left_frame)
                future_right_tip = executor.submit(detector.detect, right_frame)
                future_left_finger = executor.submit(
                    get_mediapipe_hand_data, left_frame
                )
                future_right_finger = executor.submit(
                    get_mediapipe_hand_data, right_frame
                )

                xy_l_tip, conf_l_tip, bbox_l_tip = future_left_tip.result()
                xy_r_tip, conf_r_tip, bbox_r_tip = future_right_tip.result()
                xy_l_finger, mp_results_left = future_left_finger.result()
                xy_r_finger, mp_results_right = future_right_finger.result()
            else:
                xy_l_tip, conf_l_tip, bbox_l_tip = detector.detect(left_frame)
                xy_r_tip, conf_r_tip, bbox_r_tip = detector.detect(right_frame)
                xy_l_finger, mp_results_left = get_mediapipe_hand_data(left_frame)
                xy_r_finger, mp_results_right = get_mediapipe_hand_data(right_frame)

            detection_time = (time.time() - detection_start) * 1000

            # Triangulation
            triangulation_start = time.time()

            X_tip = None
            if xy_l_tip is not None and xy_r_tip is not None:
                X_tip = triangulate_points(
                    xy_l_tip,
                    xy_r_tip,
                    stereo_calib,
                    units_to_m=config.units_to_meters,
                    rotation_angle_deg=config.rotation_angle_deg,
                    y_translation_m=config.y_translation_m,
                    coordinate_limits=config.coordinate_limits,
                )
                if X_tip is not None:
                    tip_trajectory.append(X_tip)

            X_finger = None
            if xy_l_finger is not None and xy_r_finger is not None:
                X_finger = triangulate_points(
                    xy_l_finger,
                    xy_r_finger,
                    stereo_calib,
                    units_to_m=config.units_to_meters,
                    rotation_angle_deg=config.rotation_angle_deg,
                    y_translation_m=config.y_translation_m,
                    coordinate_limits=config.coordinate_limits,
                )
                if X_finger is not None:
                    finger_trajectory.append(X_finger)

            triangulation_time = (time.time() - triangulation_start) * 1000

            # Calculate timing
            loop_time = time.time() - loop_start
            fps = 1.0 / loop_time if loop_time > 0 else 0

            # Print status periodically
            current_time = time.time()
            if current_time - last_print_time >= 1.0:  # Print every second
                last_point = X_tip if X_tip is not None else X_finger
                if last_point is not None:
                    console.print(
                        f"Frame {frame_count:6d} | "
                        f"X={last_point[0]:7.3f} Y={last_point[1]:7.3f} Z={last_point[2]:7.3f} | "
                        f"Tip: {len(tip_trajectory):3d} pts | Finger: {len(finger_trajectory):3d} pts | "
                        f"Det: {detection_time:.1f}ms | Tri: {triangulation_time:.1f}ms | "
                        f"FPS: {fps:.1f}"
                    )
                last_print_time = current_time

            # Create visualization
            left_frame_processed = draw_detection_overlays(
                left_frame,
                object_center=xy_l_tip,
                object_bbox=bbox_l_tip,
                finger_tip=xy_l_finger,
                mediapipe_results=mp_results_left,
            )

            right_frame_processed = draw_detection_overlays(
                right_frame,
                object_center=xy_r_tip,
                object_bbox=bbox_r_tip,
                finger_tip=xy_r_finger,
                mediapipe_results=mp_results_right,
            )

            # Assemble dashboard
            dashboard_image = assemble_dashboard_view(
                left_frame_processed,
                right_frame_processed,
                plot_handles,
                tip_trajectory,
                finger_trajectory,
            )

            # Display
            cv2.imshow("Perception Debug Dashboard", dashboard_image)

            frame_count += 1

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        console.print("Interrupted by user")

    finally:
        if executor:
            executor.shutdown()
        cap.release()
        cv2.destroyAllWindows()
        close_mediapipe_hands()
        console.print("Perception debugging session ended")


@app.command()
def debug_perception(
    camera_index: int = typer.Option(0, "--camera", help="Camera device index"),
    model_path: Optional[str] = typer.Option(
        None, "--model", help="Path to YOLO model"
    ),
    max_trajectory_points: int = typer.Option(
        150, "--max-points", help="Max trajectory points to display"
    ),
    enable_multithreading: bool = typer.Option(
        True,
        "--multithreading/--no-multithreading",
        help="Enable multithreaded processing",
    ),
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
) -> None:
    """Debug stereo vision and triangulation in real-time."""
    # Convert model_path to Path if provided
    model_path_obj = Path(model_path) if model_path else None

    debug_perception_impl(
        camera_index=camera_index,
        model_path=model_path_obj,
        max_trajectory_points=max_trajectory_points,
        enable_multithreading=enable_multithreading,
        config=config,
    )


if __name__ == "__main__":
    app()
