from typing import Literal
from pydantic import BaseModel


class LiveOptions(BaseModel):
    """Configuration for a Gemini Live session."""

    modalities: list[Literal["text", "audio"]] = ["audio"]
    """Response modalities. Currently only AUDIO is reliably supported by Gemini Live API."""

    voice: str | None = None
    """Voice name (e.g. "Puck", "Charon", "Kore", "Fenrir", "Aoede")."""

    language: str | None = None
    """Language code (e.g. "en")."""

    thinking_level: Literal["minimal", "low", "medium", "high"] | None = None
    """Thinking level for native audio models. Default is "minimal" for lowest latency."""

    auto_tool_execution: bool = True
    """Automatically execute tool calls and send responses back."""

    enable_input_transcription: bool = False
    """Enable transcription of user audio input."""

    enable_output_transcription: bool = False
    """Enable transcription of model audio output."""

    context_window_compression: bool = False
    """Enable context window compression for longer sessions."""

    vad_enabled: bool = True
    """Enable voice activity detection for automatic interruption handling."""

    session_resumption: bool = False
    """Enable session resumption for handling connection resets."""


__all__ = ["LiveOptions"]
