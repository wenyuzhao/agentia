import abc
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncGenerator, Sequence
import httpx
from agentia.llm import LLMOptions
from agentia.spec.base import ToolCallResponse
from agentia.spec.chat import Message
from agentia.spec.live import LiveChunk
from agentia.tools.tools import ToolSet
import agentia.spec as spec
from agentia.spec.stream import StreamPart

if TYPE_CHECKING:
    from agentia.history import History
    from agentia.live import LiveOptions


@dataclass
class GenerationResult:
    message: spec.AssistantMessage
    finish_reason: spec.FinishReason
    usage: spec.Usage
    provider_metadata: spec.ProviderMetadata | None


class Provider(abc.ABC):
    name: str
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
        self._model = model
        self.supported_urls = {}
        self._context_length: int | None = None

    @property
    def model(self) -> str:
        return self._model

    async def get_context_length(self) -> int:
        """Return the context length of the model, fetching and caching it on first call."""
        if self._context_length is None:
            self._context_length = await self._fetch_context_length()
        return self._context_length

    @abc.abstractmethod
    async def _fetch_context_length(self) -> int:
        """Fetch the context length from the provider API. Implemented by each provider."""
        ...

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
        history: "History",
    ) -> None:
        raise NotImplementedError("This provider does not support live sessions")

    async def disconnect_live(self) -> None:
        raise NotImplementedError("This provider does not support live sessions")

    async def send_live_chunk(self, chunk: LiveChunk) -> None:
        raise NotImplementedError("This provider does not support live sessions")

    async def send_tool_responses(self, responses: list[ToolCallResponse]) -> None:
        raise NotImplementedError("This provider does not support live sessions")

    async def receive(self) -> AsyncGenerator[StreamPart, None]:
        raise NotImplementedError("This provider does not support live sessions")
        # Make this a valid async generator
        yield  # type: ignore[misc]
