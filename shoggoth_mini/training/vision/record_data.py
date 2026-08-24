"""Unified data recording tool for calibration and annotation."""

import time
from pathlib import Path
from typing import Optional

import cv2
import typer
from typing_extensions import Annotated
from rich.console import Console

from ...configs.loaders import get_perception_config
from ...perception.camera import read_oriented

app = typer.Typer(help="Record data from stereo camera for calibration or annotation")
console = Console()


def record_stereo_video(
    duration_sec: int,
    output_folder: Path,
    camera_index: int = 0,
    chunk_duration_sec: Optional[int] = None,
):
    """Record stereo video for annotation purposes.

    Args:
        duration_sec: Total recording duration in seconds
        output_folder: Output directory for video files
        camera_index: Camera device index
        chunk_duration_sec: Optional chunk duration for splitting videos
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        console.print(f"[red]Error: Cannot open camera with index {camera_index}[/red]")
        raise typer.Exit(1)

    config = get_perception_config()
    width, height = config.stereo_resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    half_width = frame_width // 2

    output_folder.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0

    chunk_index = 0
    chunk_start_time = time.time()

    def create_writers(chunk_idx: int):
        left_path = output_folder / f"left_video_{timestamp}_chunk{chunk_idx}.mp4"
        right_path = output_folder / f"right_video_{timestamp}_chunk{chunk_idx}.mp4"

        left_writer = cv2.VideoWriter(
            str(left_path), fourcc, fps, (half_width, frame_height)
        )
        right_writer = cv2.VideoWriter(
            str(right_path), fourcc, fps, (half_width, frame_height)
        )
        return left_writer, right_writer

    left_writer, right_writer = create_writers(chunk_index)

    console.print(f"Recording to: {output_folder}")
    console.print(f"Duration: {duration_sec} seconds")
    console.print("Press 'q' to stop early")

    start_time = time.time()
    while time.time() - start_time < duration_sec:
        current_time = time.time()

        # Handle chunking
        if chunk_duration_sec and (
            current_time - chunk_start_time >= chunk_duration_sec
        ):
            left_writer.release()
            right_writer.release()

            chunk_index += 1
            chunk_start_time = current_time
            left_writer, right_writer = create_writers(chunk_index)
            console.print(f"Started recording chunk {chunk_index}")

        ret, frame = read_oriented(cap)
        if not ret:
            break

        left_frame = frame[:, :half_width]
        right_frame = frame[:, half_width:]
        left_writer.write(left_frame)
        right_writer.write(right_frame)

        # Display preview
        combined_frame = frame.copy()
        cv2.putText(
            combined_frame,
            "Left",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            combined_frame,
            "Right",
            (half_width + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Stereo Camera Feed", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    left_writer.release()
    right_writer.release()
    cv2.destroyAllWindows()

    console.print("Recording completed successfully")


def capture_calibration_images(
    output_folder: Path, num_pairs: int, interval_sec: int = 3, camera_index: int = 0
):
    """Capture stereo image pairs for camera calibration.

    Args:
        output_folder: Output directory for calibration images
        num_pairs: Number of image pairs to capture
        interval_sec: Interval between captures in seconds
        camera_index: Camera device index
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        console.print(f"[red]Error: Cannot open camera with index {camera_index}[/red]")
        raise typer.Exit(1)

    console.print(f"Opened camera {camera_index} successfully")

    config = get_perception_config()
    width, height = config.stereo_resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    half_width = frame_width // 2

    if frame_width == 0 or frame_height == 0:
        console.print("Error: Could not get frame dimensions from camera")
        cap.release()
        raise typer.Exit(1)

    console.print(f"Camera resolution: {frame_width}x{frame_height}")
    console.print(f"Left/right resolution: {half_width}x{frame_height}")

    output_folder.mkdir(parents=True, exist_ok=True)
    console.print(f"Saving images to: {output_folder}")

    pair_index = 1
    last_capture_time = time.time() - interval_sec

    try:
        with typer.progressbar(range(num_pairs), label="Capturing images") as progress:
            while pair_index <= num_pairs:
                current_time = time.time()

                ret, combined_frame = read_oriented(cap)
                if not ret:
                    console.print(
                        f"Error: Failed to grab frame from camera {camera_index}"
                    )
                    break

                # Show preview
                preview_frame = combined_frame.copy()
                cv2.imshow("Stereo Camera Preview", preview_frame)

                time_since_last_capture = current_time - last_capture_time
                if time_since_last_capture >= interval_sec:
                    # Capture final frame for saving
                    ret_cap, combined_frame_cap = read_oriented(cap)
                    if not ret_cap:
                        console.print("Error: Failed to capture final frame for saving")
                        break

                    frame1_cap = combined_frame_cap[:, :half_width]
                    frame2_cap = combined_frame_cap[:, half_width:]

                    img_name1 = output_folder / f"camera-1-{pair_index:02d}.jpg"
                    img_name2 = output_folder / f"camera-2-{pair_index:02d}.jpg"

                    cv2.imwrite(str(img_name1), frame1_cap)
                    cv2.imwrite(str(img_name2), frame2_cap)

                    progress.update(1)

                    pair_index += 1
                    last_capture_time = time.time()

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    console.print("Quit command received. Exiting.")
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        console.print("Recording completed successfully")


def capture_calibration_images_manual(
    output_folder: Path,
    pattern: tuple[int, int],
    camera_index: int = 0,
    detect_every: int = 2,
):
    """Manually capture stereo calibration pairs with live checkerboard feedback.

    Shows a live left|right preview that runs the checkerboard detector on both
    halves each frame and reports L/R detection status, so you press SPACE to
    save only when the board is cleanly visible in BOTH views. Detection runs on
    the downscaled preview (fast, guidance only); the FULL-RES frame is saved.

    Args:
        output_folder: Output directory for calibration images
        pattern: (cols, rows) of INTERNAL corners, for the live detection overlay
        camera_index: Camera device index
        detect_every: Run detection every N frames to keep the preview smooth
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        console.print(f"[red]Error: Cannot open camera with index {camera_index}[/red]")
        raise typer.Exit(1)

    config = get_perception_config()
    width, height = config.stereo_resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    half_width = frame_width // 2
    if frame_width == 0:
        console.print("[red]Error: Could not get frame dimensions from camera[/red]")
        cap.release()
        raise typer.Exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)
    # append after any existing pairs so we never clobber earlier good captures
    existing = sorted(output_folder.glob("camera-1-*.jpg"))
    pair_index = len(existing) + 1
    console.print(f"Camera {frame_width}x{frame_height} -> half {half_width}x{frame_height}")
    console.print(f"Detecting a {pattern[0]}x{pattern[1]} internal-corner board.")
    console.print("[bold]SPACE[/bold] = capture pair   [bold]q[/bold]/Esc = quit"
                  f"   (saving into {output_folder}, starting at {pair_index:02d})")

    sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE
    frame_i = 0
    ok_l = ok_r = False
    corners_l = corners_r = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        while True:
            ret, frame = read_oriented(cap)
            if not ret:
                console.print("[red]Error: frame grab failed[/red]")
                break
            left = frame[:, :half_width]
            right = frame[:, half_width:]

            # build a downscaled preview; detect on the preview halves so drawn
            # corners line up with what's shown (guidance only, not the saved data)
            target_w = 1600
            scale = target_w / frame_width
            preview = cv2.resize(frame, (target_w, int(frame_height * scale)))
            ph = preview.shape[1] // 2
            pv_l, pv_r = preview[:, :ph], preview[:, ph:]

            frame_i += 1
            if frame_i % detect_every == 0:
                gl = cv2.cvtColor(pv_l, cv2.COLOR_BGR2GRAY)
                gr = cv2.cvtColor(pv_r, cv2.COLOR_BGR2GRAY)
                ok_l, corners_l = cv2.findChessboardCornersSB(gl, pattern, sb_flags)
                ok_r, corners_r = cv2.findChessboardCornersSB(gr, pattern, sb_flags)

            if ok_l and corners_l is not None:
                cv2.drawChessboardCorners(pv_l, pattern, corners_l, True)
            if ok_r and corners_r is not None:
                cv2.drawChessboardCorners(pv_r, pattern, corners_r, True)

            both = ok_l and ok_r
            color = (0, 255, 0) if both else (
                (0, 180, 255) if (ok_l or ok_r) else (0, 0, 255))
            combined = cv2.hconcat([pv_l, pv_r])
            cv2.putText(combined,
                        f"L:{'OK' if ok_l else '--'}  R:{'OK' if ok_r else '--'}"
                        f"   saved:{pair_index - 1}",
                        (10, 30), font, 0.8, color, 2, cv2.LINE_AA)
            cv2.putText(combined, "SPACE=capture  q=quit",
                        (10, combined.shape[0] - 15), font, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Manual stereo capture (left | right)", combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                img1 = output_folder / f"camera-1-{pair_index:02d}.jpg"
                img2 = output_folder / f"camera-2-{pair_index:02d}.jpg"
                cv2.imwrite(str(img1), left)
                cv2.imwrite(str(img2), right)
                tag = "both OK" if both else (
                    "L only" if ok_l else ("R only" if ok_r else "no detect"))
                warn = "" if both else "  [yellow](not both — will be skipped in calib)[/yellow]"
                console.print(f"saved pair {pair_index:02d} ({tag}){warn}")
                pair_index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        console.print(f"Done. {pair_index - 1} total pairs in {output_folder}")


@app.command("annotation")
def record_annotation_video(
    duration: Annotated[int, typer.Option(help="Recording duration in seconds")] = 60,
    output_dir: Annotated[Path, typer.Option(help="Output directory")] = Path(
        "data/annotation"
    ),
    camera_index: Annotated[int, typer.Option(help="Camera index")] = 0,
    chunk_duration: Annotated[
        Optional[int], typer.Option(help="Chunk duration in seconds")
    ] = None,
):
    """Record stereo video for annotation purposes."""
    record_stereo_video(duration, output_dir, camera_index, chunk_duration)


@app.command("stereo-calibration")
def record_calibration_images(
    num_pairs: Annotated[
        int, typer.Option(help="Number of image pairs to capture (timer mode)")
    ] = 20,
    output_dir: Annotated[Path, typer.Option(help="Output directory")] = Path(
        "data/calibration"
    ),
    interval: Annotated[
        int, typer.Option(help="Interval between captures (seconds, timer mode)")
    ] = 3,
    camera_index: Annotated[int, typer.Option(help="Camera index")] = 0,
    manual: Annotated[
        bool,
        typer.Option(
            "--manual",
            help="Manual capture: press SPACE to save, with live L/R detection feedback",
        ),
    ] = False,
    cols: Annotated[
        int, typer.Option(help="Board internal corners across (for --manual overlay)")
    ] = 9,
    rows: Annotated[
        int, typer.Option(help="Board internal corners down (for --manual overlay)")
    ] = 6,
):
    """Capture stereo image pairs for camera calibration.

    Default is timed capture every --interval seconds. Use --manual to instead
    press SPACE to capture, with a live overlay showing whether the checkerboard
    is detected in BOTH halves (so you only save well-framed pairs).
    """
    if manual:
        capture_calibration_images_manual(output_dir, (cols, rows), camera_index)
    else:
        capture_calibration_images(output_dir, num_pairs, interval, camera_index)


if __name__ == "__main__":
    app()
