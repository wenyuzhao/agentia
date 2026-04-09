from typing import AsyncGenerator, Literal, TYPE_CHECKING
from pydantic import BaseModel, Field
from agentia.llm.agentic import run_agent_loop_live
from agentia.spec import StreamPart
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from agentia.agent import Agent


class LiveOptions(BaseModel):
    """Configuration for a Gemini Live session."""

    modalities: list[Literal["text", "audio"]] = Field(
        default_factory=lambda: ["audio"]
    )
    """Response modalities. Currently only AUDIO is reliably supported by Gemini Live API."""

    voice: str | None = None
    """Voice name (e.g. "Puck", "Charon", "Kore", "Fenrir", "Aoede")."""

    language: str | None = None
    """Language code (e.g. "en")."""

    thinking_level: Literal["minimal", "low", "medium", "high"] | None = None
    """Thinking level for native audio models. Default is "minimal" for lowest latency."""

    vad_enabled: bool = True
    """Enable voice activity detection for automatic interruption handling."""


class Live:
    def __init__(self, agent: "Agent", options: LiveOptions | None = None):
        self.agent = agent
        self.options = options or LiveOptions()

    async def __aenter__(self):
        await self.agent.__aenter__()
        # Connect live session if provider supports it
        assert self.agent.provider.supports_live
        await self.agent.tools.init()
        instructions = self.agent.history.get_instructions() or None
        await self.agent.provider.connect_live(
            self.options, self.agent.tools, instructions, history=self.agent.history
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Disconnect live session if provider supports it
        await self.agent.provider.disconnect_live()
        await self.agent.__aexit__(exc_type, exc_val, exc_tb)

    async def send_audio(
        self, data: bytes, mime_type: str = "audio/pcm;rate=16000"
    ) -> None:
        """Send an audio chunk to the live session. Default: PCM 16kHz 16-bit mono."""
        await self.agent.provider.send_audio(data, mime_type)

    async def send_video(self, data: bytes, mime_type: str = "image/jpeg") -> None:
        """Send a video frame to the live session."""
        await self.agent.provider.send_video(data, mime_type)

    async def send_text(self, text: str) -> None:
        """Send text input to the live session."""
        await self.agent.provider.send_text_live(text)

    async def send_audio_stream_end(self) -> None:
        """Signal end of audio stream to flush cached audio."""
        await self.agent.provider.send_audio_stream_end()

    async def receive(self) -> AsyncGenerator[StreamPart, None]:
        """Receive stream parts from the live session."""
        async for event in run_agent_loop_live(self.agent):
            yield event


__all__ = ["LiveOptions", "Live"]
