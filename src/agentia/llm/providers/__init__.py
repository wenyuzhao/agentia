import abc
import re
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field

import httpx

from agentia.llm import LLMOptions
from agentia.models import Instruction, Item, ToolSchema
from agentia.models.stream import StreamEvent


class LLMError(Exception):
    """A failed model call, after retries and fallbacks."""

    def __init__(
        self, message: str, status: int | None = None, retryable: bool = False
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass
class ModelRequest:
    """One stateless model call: the whole context is sent every time."""

    items: Sequence[Item]
    instructions: Sequence[Instruction] = ()
    summary: str | None = None
    """The latest compaction summary; placed between static and dynamic instructions."""
    tools: Sequence[ToolSchema] = ()
    options: LLMOptions = field(default_factory=LLMOptions)


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
        self.__context_length: int | None = None

    @property
    def model(self) -> str:
        return self._model

    @abc.abstractmethod
    def with_model(self, model: str) -> "Provider":
        """The same provider (credentials, endpoint, client) for another model."""

    async def get_context_length(self) -> int:
        """Return the context length of the model, fetching and caching it on first call."""
        if self.__context_length is None:
            self.__context_length = await self._fetch_context_length()
        return self.__context_length

    @abc.abstractmethod
    async def _fetch_context_length(self) -> int:
        """Fetch the context length from the provider API. Implemented by each provider."""
        ...

    @abc.abstractmethod
    def stream(
        self, request: ModelRequest, client: httpx.AsyncClient | None = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream one response. The terminal `response.completed` / `.incomplete` event
        carries the complete output items. Raises `LLMError` on failure."""
        ...


__all__ = ["LLMError", "ModelRequest", "Provider"]
