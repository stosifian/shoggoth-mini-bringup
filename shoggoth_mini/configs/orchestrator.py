"""Orchestrator configuration for the main application."""

import os
import logging
from pydantic import Field
from .base import BaseConfig

logger = logging.getLogger(__name__)


class OrchestratorConfig(BaseConfig):
    """Configuration for the orchestrator application."""

    # OpenAI API Configuration
    openai_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="OpenAI API key for Realtime API access (loads from OPENAI_API_KEY env var)",
    )
    websocket_url: str = Field(
        default="wss://api.openai.com/v1/realtime?model=gpt-realtime",
        description="WebSocket URL for OpenAI Realtime API (GA)",
    )
    websocket_headers: list[str] = Field(
        default_factory=lambda: [
            "Authorization: Bearer YOUR_API_KEY_HERE",
        ],
        description="WebSocket headers for OpenAI Realtime API connection (GA: no OpenAI-Beta header)",
    )

    # Audio Configuration
    audio_sample_rate: int = Field(
        default=16_000, description="Audio sample rate in Hz"
    )
    audio_channels: int = Field(default=1, description="Number of audio channels")
    audio_dtype: str = Field(default="int16", description="Audio data type")
    audio_block_size: int = Field(
        default=2_048, description="Audio block size for processing"
    )

    # Visual Configuration
    # Gaussian noise added to every cursor command inside the motion primitives.
    # Upstream defaults to 0.010, which is ~41 ticks of sigma per motor and, taken
    # over the many commands a session issues, reaches ~200 ticks in the tail —
    # enough to push the deepest primitive past the encoder range and have it
    # refused. Set to 0.0 for deterministic, repeatable motion while testing.
    motion_noise_scale: float = Field(
        default=0.0,
        ge=0.0,
        description="Cursor noise for motion primitives. 0 disables randomisation.",
    )

    wave_detection_depth_z_max: float = Field(
        default=-0.40,
        description="Depth threshold for wave detection (Z distance)",
    )

    # Finger-following configuration
    finger_follow_z_threshold: float = Field(
        default=-0.14,
        description="Depth threshold (Z) that defines when a finger is considered 'near' for finger-following",
    )
    finger_follow_y_threshold: float = Field(
        default=0.25,
        description="Maximum Y height threshold for finger-following (finger must be below this height)",
    )

    # Timing Configuration
    idle_start_delay_seconds: float = Field(
        default=2.0, description="Delay before starting idle motion after speech stops"
    )

    # System Messages and Tools
    system_prompt: str = Field(
        description="System prompt for the AI assistant",
    )

    def get_websocket_headers(self) -> list[str]:
        """Get WebSocket headers with proper API key integration.

        GA Realtime API: only Bearer auth; the beta `OpenAI-Beta: realtime=v1`
        header was removed when the beta interface was retired (2026-05-12).
        """
        return [
            f"Authorization: Bearer {self.openai_api_key}",
        ]

    def get_session_config(self) -> dict:
        """Build the GA `session.update` `session` object.

        GA schema differs from the old beta: a required `type: "realtime"`,
        `output_modalities` (was `modalities`), and turn detection nested under
        `audio.input.turn_detection` (was top-level `turn_detection`). This
        assistant is voice-in / text+action-out, so output is text only.
        """
        return {
            "type": "realtime",
            "output_modalities": ["text"],
            "instructions": self.system_prompt,
            "tools": self.get_tools_definition(),
            "audio": {
                "input": {
                    "turn_detection": {
                        "type": "server_vad",
                        "interrupt_response": False,
                        "create_response": False,
                    },
                },
            },
        }

    def get_tools_definition(self) -> list:
        """Get tool definitions for OpenAI function calling."""
        return [
            {
                "type": "function",
                "name": "perform_primitive",
                "description": (
                    "Performs a motion primitive to express the assistant's current physical state in response to the user's input. "
                    "Available primitives: <yes>, <no>, <shake>, <circle>, <grab_object>, <release_object>, <high_five>"
                    "Use <yes> for agreement/understanding, eg. when user asks you a question. "
                    "Use <no> for disagreement/confusion, eg. when user asks you a question. "
                    "Use <shake> for waving your hand, to say hi or goodbye or similar. "
                    "Use <circle> for expressing excitement or happiness. "
                    "Use <grab_object> when the user asks you to grab or hold his finger or an object. "
                    "Use <release_object> when the user asks you to release. Do not release if the user is not asking you to release explicitly. "
                    "Use <high_five> when the user asks for a high five or when celebrating something together."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "The primitive token to execute (<yes>, <no>, <shake>, <circle>, <grab_object>, <release_object>).",
                            "enum": [
                                "<yes>",
                                "<no>",
                                "<shake>",
                                "<circle>",
                                "<grab_object>",
                                "<release_object>",
                                "<high_five>",
                            ],
                        },
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "stay_silent",
                "description": "Use this function to give the user an opportunity to finish their thought.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "type": "function",
                "name": "follow_finger",
                "description": (
                    "Starts a closed-loop policy that makes the robot continuously track the user's fingertip until stopped. "
                    "Call this when you receive a <finger near> visual cue. Do not call it if the user is asking you to grab or hold something, this is a separate tool."
                    "Do not call it if the policy is already running."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        ]
