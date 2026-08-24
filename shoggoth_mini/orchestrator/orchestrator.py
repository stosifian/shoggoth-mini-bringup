"""Main orchestrator application with async architecture."""

import asyncio
import json
import base64
import time
from typing import Optional, Dict
import cv2
import collections as py_collections
import sounddevice as sd
import websockets
import logging
from rich.console import Console

from ..configs.orchestrator import OrchestratorConfig
from ..configs.hardware import HardwareConfig
from ..configs.perception import PerceptionConfig
from ..configs.control import ControlConfig

from ..hardware.motors import MotorController
from ..control.primitives import execute_behavior, MotionBehavior
from ..control.idle import IdleMotionLoop
from ..perception.hand_tracking import (
    get_mediapipe_hand_data,
    update_landmark_trail,
    is_wave_gesture,
    close_mediapipe_hands,
    INDEX_FINGER_MCP,
)
from ..perception.stereo import (
    triangulate_points,
    load_stereo_calibration,
    StereoCalibration,
    split_stereo_frame,
)
from ..control.closed_loop import ClosedLoopController
from ..perception.camera import read_oriented

logger = logging.getLogger(__name__)
console = Console()


class OrchestratorApp:
    """Main orchestrator application coordinating all system components."""

    def __init__(
        self,
        orchestrator_config: Optional[OrchestratorConfig] = None,
        hardware_config: Optional[HardwareConfig] = None,
        perception_config: Optional[PerceptionConfig] = None,
        control_config: Optional[ControlConfig] = None,
    ):
        """Initialize orchestrator application."""
        self.config = orchestrator_config or OrchestratorConfig()
        self.hardware_config = hardware_config or HardwareConfig()
        self.perception_config = perception_config or PerceptionConfig()
        self.control_config = control_config or ControlConfig()

        # WebSocket connection
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.ws_connected = asyncio.Event()

        # Audio queues
        self.audio_input_queue: asyncio.Queue = asyncio.Queue()
        self.visual_input_queue: asyncio.Queue = asyncio.Queue()

        # System state
        self.is_recording = asyncio.Event()
        self.is_recording.set()
        self.is_visual_worker_running = asyncio.Event()
        self.is_grabbing = asyncio.Event()

        # Motor controller and idle motion
        self.motor_controller: Optional[MotorController] = None
        self.idle_motion_loop: Optional[IdleMotionLoop] = None
        self.idle_start_task: Optional[asyncio.Task] = None

        # Active function calls tracking
        self.active_function_calls: Dict[str, Dict] = {}

        # Stereo vision components
        self.stereo_calibration: StereoCalibration = None
        self._init_stereo_vision()

        # Event loop reference for thread-safe callbacks
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Active finger-follow controller
        self.finger_follow_controller: Optional[ClosedLoopController] = None
        self.finger_follow_thread: Optional[asyncio.Future] = (
            None  # thread run via to_thread
        )
        self.is_finger_following = asyncio.Event()

    def _init_stereo_vision(self) -> None:
        """Initialize stereo vision components."""
        try:
            self.stereo_calibration = load_stereo_calibration()
            logger.info("Successfully loaded stereo calibration data.")
        except Exception as e:
            logger.warning(f"Warning: Could not load stereo calibration: {e}")

    async def start(self) -> None:
        """Start the orchestrator application."""
        logger.info("Starting Shoggoth Mini...")
        # Store running loop reference
        self.loop = asyncio.get_running_loop()

        # Initialize motor controller
        try:
            self.motor_controller = MotorController(self.hardware_config)
            await asyncio.to_thread(self.motor_controller.connect)
            logger.info("Motor controller connected successfully.")
        except Exception as e:
            logger.warning(f"Warning: Could not connect motor controller: {e}")

        # Start all async tasks
        tasks = [
            asyncio.create_task(self._audio_input_worker()),
            asyncio.create_task(self._websocket_worker()),
            asyncio.create_task(self._audio_send_worker()),
        ]

        # Add visual processing
        self.is_visual_worker_running.set()
        tasks.extend(
            [
                asyncio.create_task(self._visual_processing_worker()),
                asyncio.create_task(self._visual_input_send_worker()),
            ]
        )
        logger.info("Visual processing (wave detection) enabled.")

        try:
            # Wait for all tasks
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("\nKeyboardInterrupt - shutting down")
        finally:
            await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up...")

        # Stop visual processing
        self.is_visual_worker_running.clear()

        # Stop idle motion
        await self._stop_idle_motion(reset_to_calibrated=False)

        # Stop the finger-follow controller BEFORE the bus is closed. Previously
        # this ran after disconnect(), so quitting mid-follow left the controller
        # shutting down against an already-closed port.
        await self._stop_finger_follow()

        # Cancel pending tasks
        if self.idle_start_task and not self.idle_start_task.done():
            self.idle_start_task.cancel()

        self.is_grabbing.clear()

        # Return to neutral UNCONDITIONALLY. This used to happen only via the idle
        # path, so whether the tentacle came home depended on which state happened
        # to be active at Ctrl+C — idle running meant a reset, a just-finished
        # primitive meant none, and the servos held their last pose because torque
        # stays enabled through disconnect(). Ramped, so it is a smooth return.
        if self.motor_controller:
            try:
                await asyncio.to_thread(
                    self.motor_controller.reset_to_calibrated_positions
                )
            except Exception as e:
                logger.warning("Could not return to neutral on shutdown: %s", e)

        # Disconnect motor controller
        if self.motor_controller:
            await asyncio.to_thread(self.motor_controller.disconnect)

        # Close MediaPipe resources
        close_mediapipe_hands()

        self.is_recording.clear()

        logger.info("Cleanup complete.")

    async def _websocket_worker(self) -> None:
        """Worker for WebSocket communication with OpenAI."""
        uri = self.config.websocket_url
        headers = dict(h.split(": ", 1) for h in self.config.get_websocket_headers())

        logger.info(f"Connecting to WebSocket: {uri}")
        logger.debug(f"Headers: {headers}")

        try:
            async with websockets.connect(uri, additional_headers=headers) as websocket:
                self.websocket = websocket

                logger.info("WebSocket connected. Waiting for session.created...")

                async for message in websocket:
                    await self._handle_websocket_message(message)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.ws_connected.clear()

    async def _handle_websocket_message(self, message: str) -> None:
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            event_type = data.get("type")

            # Log OpenAI messages to debug issues
            logger.debug(f"OpenAI message: {event_type} - {data}")

            if event_type == "session.created":
                await self._handle_session_created()
            elif event_type == "session.updated":
                logger.info("Session updated - ready.")
                self.ws_connected.set()
            elif event_type == "input_audio_buffer.speech_started":
                await self._handle_speech_started()
            elif event_type == "input_audio_buffer.speech_stopped":
                await self._handle_speech_stopped()
            elif event_type == "input_audio_buffer.committed":
                await self._handle_audio_committed()
            elif event_type == "response.output_text.delta":
                self._handle_text_output(data.get("delta", ""), end="")
            elif event_type == "response.output_text.done":
                self._handle_text_output(" ← [text complete]", end="\n")
            elif event_type == "response.created":
                logger.debug("Response creation started")
            elif event_type == "response.done":
                await self._handle_response_completion()
            elif event_type == "conversation.item.created":
                logger.debug("Conversation item created")
            elif event_type == "response.output_item.added":
                await self._handle_function_item_added(data)
            elif event_type == "response.function_call_arguments.delta":
                await self._handle_function_arguments_delta(data)
            elif event_type == "response.function_call_arguments.done":
                await self._handle_function_arguments_done(data)
            elif event_type == "error":
                await self._handle_api_error(data)
            else:
                logger.debug(f"Unhandled event type: {event_type}")

        except Exception as exc:
            logger.exception("Message handler error: %s", exc)

    def _handle_text_output(self, text: str, end: str = "\n") -> None:
        """Handle text output to console."""
        console.print(text, end=end)

    async def _handle_response_completion(self) -> None:
        """Handle response completion and restart idle motion if appropriate."""
        logger.info("Response completed")
        # Restart idle motion after response completes (but not if we're in grab state or finger following)
        if (
            self.is_recording.is_set()
            and not self.is_grabbing.is_set()
            and not self.is_finger_following.is_set()
        ):
            await self._start_idle_motion()

    async def _handle_session_created(self) -> None:
        """Handle session created event."""
        logger.info("Session created. Updating session...")
        update = {
            "type": "session.update",
            "session": self.config.get_session_config(),
        }
        await self._send_websocket_message(update)

        # Send greeting (output modality is set on the session -> text)
        greeting = {"type": "response.create"}
        await self._send_websocket_message(greeting)

        # Start idle motion
        await self._schedule_idle_start()

    async def _handle_speech_started(self) -> None:
        """Handle speech started event."""
        logger.info("[VAD] Speech started")
        await self._stop_idle_motion(reset_to_calibrated=True)
        if self.idle_start_task and not self.idle_start_task.done():
            self.idle_start_task.cancel()
            logger.info("Cancelled pending idle start due to speech.")

    async def _handle_speech_stopped(self) -> None:
        """Handle speech stopped event."""
        logger.info("[VAD] Speech stopped")
        if not self.is_grabbing.is_set():
            await self._schedule_idle_start()

    async def _handle_audio_committed(self) -> None:
        """Handle audio committed event."""
        logger.info("[VAD] Audio committed; requesting text response...")

        # Cancel pending idle start since we're expecting a response
        if self.idle_start_task and not self.idle_start_task.done():
            self.idle_start_task.cancel()
            logger.info("Cancelled pending idle start due to audio commit.")

        request = {"type": "response.create"}
        logger.info("Sending response.create request to OpenAI...")
        await self._send_websocket_message(request)
        logger.info("Response request sent successfully")

    async def _handle_function_item_added(self, data: dict) -> None:
        """Handle function call item added."""
        item = data.get("item", {})
        if item.get("type") == "function_call":
            fid = item["id"]
            self.active_function_calls[fid] = {
                "name": item["name"],
                "call_id": item["call_id"],
                "arguments": "",
            }

    async def _handle_function_arguments_delta(self, data: dict) -> None:
        """Handle function call arguments delta."""
        fid = data.get("item_id")
        if fid in self.active_function_calls:
            self.active_function_calls[fid]["arguments"] += data.get("delta", "")

    async def _handle_function_arguments_done(self, data: dict) -> None:
        """Handle function call arguments completion."""
        fid = data.get("item_id")
        if fid in self.active_function_calls:
            info = self.active_function_calls.pop(fid)
            await self._execute_function_call(info)

    async def _handle_api_error(self, data: dict) -> None:
        """Handle API error responses."""
        logger.error(f"API error received: {data}")

        error_details = data.get("error", {})
        error_message = error_details.get("message")

        if error_message == "Conversation already has an active response":
            logger.warning("OpenAI API is busy with active response")

    async def _execute_function_call(self, info: dict) -> None:
        """Execute function call and return result."""
        name = info["name"]
        args = info["arguments"]
        call_id = info["call_id"]

        logger.info(f"Executing function: {name} with args: {args}")

        # Stop idle motion and finger following for actions
        await self._stop_idle_motion(reset_to_calibrated=False)
        if self.idle_start_task and not self.idle_start_task.done():
            self.idle_start_task.cancel()

        # Stop finger following before executing other actions (except follow_finger itself)
        if name != "follow_finger" and self.is_finger_following.is_set():
            await self._stop_finger_follow()
            logger.info("Stopped finger following to execute other action.")

        result = {"status": "error", "message": "unknown tool"}
        should_restart_idle = True

        if name == "perform_primitive":
            # Handle perform_primitive function call directly
            try:
                payload = json.loads(args)
                action_str = payload["action"]

                if not self.motor_controller:
                    logger.warning("%s (Motor controller not available)", action_str)
                    result = {
                        "status": "success",
                        "action": action_str,
                        "message": "Motor controller not available",
                    }
                else:
                    # Get behavior from action string
                    behavior = MotionBehavior.from_action_string(action_str)

                    if behavior:
                        logger.info("Performing physical action: %s", action_str)

                        # Execute behavior using the refactored primitives
                        motor_result = await asyncio.to_thread(
                            execute_behavior, self.motor_controller, behavior
                        )

                        # Handle grab/release state management without string literals
                        if behavior == MotionBehavior.GRAB:
                            self.is_grabbing.set()
                            should_restart_idle = False
                            logger.info(
                                "Holding grab position - idle motion will not restart"
                            )
                        elif behavior == MotionBehavior.RELEASE:
                            self.is_grabbing.clear()

                        result = {
                            "status": "success",
                            "action": action_str,
                            "motor_result": motor_result,
                        }
                    else:
                        logger.warning("Unknown action: %s", action_str)
                        result = {
                            "status": "error",
                            "message": f"Unknown action: {action_str}",
                        }

            except Exception as exc:
                logger.error(f"Error executing action: {exc}")
                result = {"status": "error", "message": str(exc)}

        elif name == "follow_finger":
            await self._start_finger_follow()
            result = {"status": "success"}
            should_restart_idle = False  # finger-follow handles its own motion
        elif name == "stay_silent":
            logger.info("Staying silent to let user complete their thought...")
            result = {"status": "success"}

        # Send result back to API
        await self._send_function_result(call_id, result)

        # Request follow-up response
        request = {"type": "response.create"}
        await self._send_websocket_message(request)

        # Schedule idle restart only for actions that should return to neutral
        if should_restart_idle and self.is_recording.is_set():
            await self._schedule_idle_start()

    async def _schedule_idle_start(self) -> None:
        """Schedule idle motion to start after delay."""
        if self.idle_start_task and not self.idle_start_task.done():
            self.idle_start_task.cancel()

        logger.info(
            f"Scheduling idle motion to start in {self.config.idle_start_delay_seconds}s."
        )
        self.idle_start_task = asyncio.create_task(self._start_idle_after_delay())

    async def _start_idle_after_delay(self) -> None:
        """Start idle motion after configured delay."""
        await asyncio.sleep(self.config.idle_start_delay_seconds)
        await self._start_idle_motion()

    async def _start_idle_motion(self) -> None:
        """Start idle motion if not already running."""
        if not self.motor_controller:
            return

        if self.idle_motion_loop and self.idle_motion_loop.is_running:
            return

        logger.info("Starting idle motion")

        if not self.idle_motion_loop:
            self.idle_motion_loop = IdleMotionLoop(
                motor_controller=self.motor_controller,
                pattern_config=self.control_config.idle_motion_pattern_config,
                hz=self.hardware_config.idle_motion_hz,
                max_motor_step_per_loop=self.control_config.idle_motion_max_motor_step_per_loop,
            )

        self.idle_motion_loop.start()

    async def _stop_idle_motion(self, reset_to_calibrated: bool = True) -> None:
        """Stop idle motion if running."""
        if self.idle_motion_loop and self.idle_motion_loop.is_running:
            logger.info("Stopping idle motion")
            self.idle_motion_loop.stop(reset_to_calibrated=reset_to_calibrated)

    async def _audio_input_worker(self) -> None:
        """Worker for audio input capture."""
        logger.info("Audio input worker started.")

        def callback(indata, _frames, _time, status):
            if status:
                logger.error("Audio input status: %s", status)
            if self.is_recording.is_set():
                # Put audio data into the asyncio queue from this IO thread safely
                if self.loop and not self.loop.is_closed():
                    try:
                        self.loop.call_soon_threadsafe(
                            self.audio_input_queue.put_nowait, bytes(indata)
                        )
                    except RuntimeError:
                        pass

        try:
            with sd.InputStream(
                samplerate=self.config.audio_sample_rate,
                blocksize=self.config.audio_block_size,
                dtype=self.config.audio_dtype,
                channels=self.config.audio_channels,
                callback=callback,
            ) as stream:
                logger.info(
                    f"Mic stream open: SR={stream.samplerate}, Ch={stream.channels}"
                )
                while self.is_recording.is_set():
                    await asyncio.sleep(0.1)
        except Exception as exc:
            logger.exception("Audio input error: %s", exc)
        finally:
            logger.info("Audio input worker finished.")

    async def _audio_send_worker(self) -> None:
        """Worker for sending audio to WebSocket."""
        try:
            await asyncio.wait_for(self.ws_connected.wait(), timeout=20)
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for websocket ready for audio sender")
            return

        logger.info("Audio send worker started.")

        while self.ws_connected.is_set() and self.is_recording.is_set():
            try:
                chunk = await asyncio.wait_for(
                    self.audio_input_queue.get(), timeout=0.05
                )
                if chunk:
                    await self._send_audio(chunk)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.error("audio_send_worker error:", exc)
                await asyncio.sleep(0.1)

        logger.info("Audio send worker finished.")

    async def _visual_input_send_worker(self) -> None:
        """Worker for sending visual input to WebSocket."""
        try:
            await asyncio.wait_for(self.ws_connected.wait(), timeout=20)
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out waiting for websocket ready for visual input sender"
            )
            return

        logger.info("Visual input send worker started.")

        while self.ws_connected.is_set() and self.is_visual_worker_running.is_set():
            try:
                visual_description = await asyncio.wait_for(
                    self.visual_input_queue.get(), timeout=0.1
                )
                if visual_description:
                    await self._send_visual_input(visual_description)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.error(f"Visual input send worker error: {exc}")
                await asyncio.sleep(0.1)

        logger.info("Visual input send worker finished.")

    async def _visual_processing_worker(self) -> None:
        """Worker for visual processing (wave detection only)."""
        logger.info("Visual processing worker starting...")

        cap_stereo = None

        try:
            # Initialize stereo camera
            cap_stereo = cv2.VideoCapture(self.perception_config.camera_index)
            if not cap_stereo.isOpened():
                logger.error(
                    f"ERROR: Cannot open stereo camera {self.perception_config.camera_index}"
                )
                return

            cap_stereo.set(
                cv2.CAP_PROP_FRAME_WIDTH, self.perception_config.stereo_resolution[0]
            )
            cap_stereo.set(
                cv2.CAP_PROP_FRAME_HEIGHT, self.perception_config.stereo_resolution[1]
            )

            actual_fw = int(cap_stereo.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_fh = int(cap_stereo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Stereo camera opened: {actual_fw}x{actual_fh}")

            # Wave detection variables
            trail_wave = py_collections.deque(maxlen=30)
            last_hand_seen_time_wave = None
            HAND_ABSENCE_TIMEOUT_WAVE = 1.0
            last_near_sent_time = 0.0
            LANDMARK_FOR_WAVE = INDEX_FINGER_MCP

            # Main processing loop
            while self.is_visual_worker_running.is_set():
                ok_stereo, frame_stereo_full = read_oriented(cap_stereo)
                if not ok_stereo or frame_stereo_full is None:
                    await asyncio.sleep(0.1)
                    continue

                left_frame, right_frame = split_stereo_frame(frame_stereo_full)

                if left_frame.size == 0 or right_frame.size == 0:
                    await asyncio.sleep(0.1)
                    continue

                current_time = time.time()

                # Get hand data from both frames (includes MediaPipe processing)
                xy_l_finger, mp_results_left = get_mediapipe_hand_data(left_frame)
                xy_r_finger, mp_results_right = get_mediapipe_hand_data(right_frame)

                if xy_l_finger is not None and xy_r_finger is not None:

                    try:

                        # Update wave detection trail (using left frame results only is enough) and detect wave
                        last_hand_seen_time_wave, _, _ = update_landmark_trail(
                            trail_wave,
                            mp_results_left,
                            LANDMARK_FOR_WAVE,
                            last_hand_seen_time_wave,
                            current_time,
                            HAND_ABSENCE_TIMEOUT_WAVE,
                        )
                        wave_detected, _ = is_wave_gesture(trail_wave)

                        # Triangulate points for depth information
                        point_3d = triangulate_points(
                            xy_l_finger,
                            xy_r_finger,
                            self.stereo_calibration,
                            units_to_m=self.perception_config.units_to_meters,
                            rotation_angle_deg=self.perception_config.rotation_angle_deg,
                            y_translation_m=self.perception_config.y_translation_m,
                            coordinate_limits=self.perception_config.coordinate_limits,
                        )

                        if point_3d is not None:
                            height_y, depth_z = point_3d[1], point_3d[2]
                            logger.debug(f"Depth: {depth_z}, Height: {height_y}")
                            # Check for wave detection and bounds
                            if (
                                depth_z < self.config.wave_detection_depth_z_max
                                and wave_detected
                            ):
                                if wave_detected:
                                    logger.info(
                                        "[VISUAL] Hand wave detected (Z=%.2f, Y=%.2f)",
                                        depth_z,
                                        height_y,
                                    )
                                    await self.visual_input_queue.put("<user waving>")
                                    trail_wave.clear()
                                    await asyncio.sleep(0.5)

                            # Only notify LLM if we are not currently following and we haven't spammed recently
                            # and the finger is close enough (Z) and low enough (Y)
                            if (
                                depth_z > self.config.finger_follow_z_threshold
                                and height_y < self.config.finger_follow_y_threshold
                                and not self.is_finger_following.is_set()
                                and current_time - last_near_sent_time > 1.0
                            ):
                                last_near_sent_time = current_time
                                logger.info(
                                    "[VISUAL] Finger near (Z=%.2f, Y=%.2f)",
                                    depth_z,
                                    height_y,
                                )
                                await self.visual_input_queue.put("<finger near>")

                    except Exception as e:
                        logger.error(f"Error during detection: {e}")

                await asyncio.sleep(0.01)

        except Exception as exc:
            logger.exception("Visual processing worker error: %s", exc)
        finally:
            if cap_stereo:
                cap_stereo.release()
            close_mediapipe_hands()
            logger.info("Visual processing worker finished.")

    async def _send_audio(self, chunk: bytes) -> None:
        """Send audio chunk to WebSocket."""
        if self.websocket and chunk and self.ws_connected.is_set():
            try:
                message = {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode(),
                }
                await self.websocket.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending audio chunk: {e}")

    async def _send_visual_input(self, description: str) -> None:
        """Send visual input to WebSocket."""
        if self.websocket and description and self.ws_connected.is_set():
            logger.info("Sending visual input to AI: %s", description)
            try:
                # Send visual cue as user message
                message = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": description}],
                    },
                }
                await self.websocket.send(json.dumps(message))

                # Request response (output modality set on the session -> text)
                response_request = {"type": "response.create"}
                await self.websocket.send(json.dumps(response_request))
            except Exception as e:
                logger.error(f"Error sending visual input: {e}")

    async def _send_websocket_message(self, message: dict) -> None:
        """Send message to WebSocket."""
        if self.websocket:
            try:
                message_json = json.dumps(message)
                logger.debug(
                    f"Sending WebSocket message: {message.get('type', 'unknown')}"
                )
                await self.websocket.send(message_json)
            except Exception as e:
                logger.error(f"Failed to send WebSocket message: {e}")
                logger.error(f"Message type: {message.get('type', 'unknown')}")
                raise

    async def _send_function_result(self, call_id: str, result: dict) -> None:
        """Send function call result back to API."""
        message = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result),
            },
        }
        await self._send_websocket_message(message)

    async def _start_finger_follow(self) -> None:
        """Start the closed-loop finger-following controller in a background thread."""
        if self.is_finger_following.is_set():
            return

        # Stop idle motion and visual worker to free camera resources
        await self._stop_idle_motion(reset_to_calibrated=False)
        if self.idle_start_task and not self.idle_start_task.done():
            self.idle_start_task.cancel()

        # Stop visual processing worker to free the stereo camera for the closed-loop controller.
        if self.is_visual_worker_running.is_set():
            self.is_visual_worker_running.clear()

        # Instantiate controller if not existing
        if not self.finger_follow_controller:
            self.finger_follow_controller = ClosedLoopController(
                control_config=self.control_config,
                perception_config=self.perception_config,
                hardware_config=self.hardware_config,
                external_motor_controller=self.motor_controller,
            )

        # Run in background thread using asyncio.to_thread (returns coroutine) and schedule
        self.finger_follow_thread = asyncio.create_task(
            asyncio.to_thread(self.finger_follow_controller.start)
        )

        # Call _stop_finger_follow when the thread completes (when prolonged loss detected by the controller)
        self.finger_follow_thread.add_done_callback(
            lambda fut: asyncio.create_task(self._stop_finger_follow())
        )
        self.is_finger_following.set()
        logger.info("Finger-following controller started")

    async def _stop_finger_follow(self) -> None:
        """Stop the finger-following controller and resume normal behaviour."""
        if not self.is_finger_following.is_set():
            return

        if self.finger_follow_controller:
            try:
                self.finger_follow_controller.stop(
                    reset_to_calibrated=False
                )  # To avoid unintentional motor movement
            except Exception as exc:
                logger.error(f"Error stopping finger follower: {exc}")

        # Await thread completion if we have a Future
        if self.finger_follow_thread:
            try:
                await asyncio.wait_for(self.finger_follow_thread, timeout=5.0)
            except Exception:
                pass
            self.finger_follow_thread = None

        self.is_finger_following.clear()
        # TODO: avoid resetting the controller each time
        self.finger_follow_controller = None  # allow garbage collection / fresh init
        logger.info("Finger-following controller stopped")

        # Inform LLM that finger following has ended so it can decide next action
        await self.visual_input_queue.put("<finger follow stopped>")

        # Restart visual worker regardless of auto/manual so that wave & proximity cues resume
        if not self.is_visual_worker_running.is_set():
            asyncio.create_task(self._restart_visual_worker_after_delay())

        # Resume idle motion
        if self.is_recording.is_set():
            await self._schedule_idle_start()

    async def _restart_visual_worker_after_delay(self, delay: float = 0.35) -> None:
        """Helper to restart visual worker after brief delay to ensure camera is free."""
        await asyncio.sleep(delay)
        if not self.is_visual_worker_running.is_set():
            self.is_visual_worker_running.set()
            asyncio.create_task(self._visual_processing_worker())
            asyncio.create_task(self._visual_input_send_worker())
