import abc
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncGenerator, Sequence
import httpx
from agentia.llm import LLMOptions
from agentia.spec.chat import Message
from agentia.tools.tools import ToolSet
import agentia.spec as spec
from agentia.spec.stream import StreamPart

if TYPE_CHECKING:
    from agentia.live import LiveOptions


@dataclass
class GenerationResult:
    message: spec.AssistantMessage
    finish_reason: spec.FinishReason
    usage: spec.Usage
    provider_metadata: spec.ProviderMetadata | None


class Provider(abc.ABC):
    name: str
    model: str
    supported_urls: dict[str, Sequence[re.Pattern | str]]
    """
    Supported URL patterns by media type for the provider.

    The keys are media type patterns or full media types (e.g. `*\\/*` for everything, `audio/*`, `video/*`, or `application/pdf`).
    and the values are arrays of regular expressions that match the URL paths.

    The matching should be against lower-case URLs.

    Matched URLs are supported natively by the model and are not downloaded.
    """

    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model
        self.supported_urls = {}

    @abc.abstractmethod
    async def generate(
        self,
        messages: list[Message],
        tools: ToolSet,
        options: LLMOptions,
        client: httpx.AsyncClient,
    ) -> GenerationResult: ...

    @abc.abstractmethod
    def stream(
        self,
        messages: list[Message],
        tools: ToolSet,
        options: LLMOptions,
        client: httpx.AsyncClient,
    ) -> AsyncGenerator[StreamPart, None]: ...

    # --- Live session methods (optional, for bidirectional providers) ---

    @property
    def supports_live(self) -> bool:
        return False

    async def connect_live(
        self,
        options: "LiveOptions",
        tools: ToolSet,
        instructions: str | None,
    ) -> None:
        raise NotImplementedError("This provider does not support live sessions")

    async def disconnect_live(self) -> None:
        raise NotImplementedError("This provider does not support live sessions")

    async def send_audio(
        self, data: bytes, mime_type: str = "audio/pcm;rate=16000"
    ) -> None:
        raise NotImplementedError("This provider does not support live sessions")

    async def send_video(self, data: bytes, mime_type: str = "image/jpeg") -> None:
        raise NotImplementedError("This provider does not support live sessions")

    async def send_text_live(self, text: str) -> None:
        raise NotImplementedError("This provider does not support live sessions")

    async def send_audio_stream_end(self) -> None:
        raise NotImplementedError("This provider does not support live sessions")

    async def send_tool_response(self, tool_call_id: str, output: object) -> None:
        raise NotImplementedError("This provider does not support live sessions")

    async def receive(self) -> AsyncGenerator[StreamPart, None]:
        raise NotImplementedError("This provider does not support live sessions")
        # Make this a valid async generator
        yield  # type: ignore[misc]
