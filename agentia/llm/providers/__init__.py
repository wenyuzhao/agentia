import abc
import re
from dataclasses import dataclass
from typing import AsyncGenerator, Sequence
import httpx
from agentia.llm import LLMOptions
from agentia.models.chat import Message
from agentia.tools.tools import ToolSet
import agentia.models as models
from agentia.models.stream import StreamPart


@dataclass
class GenerationResult:
    message: models.AssistantMessage
    finish_reason: models.FinishReason
    usage: models.Usage


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
        self.__model = model
        self.supported_urls = {}
        self.__context_length: int | None = None

    @property
    def model(self) -> str:
        return self.__model

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
    async def generate(
        self,
        instructions: str,
        messages: list[Message],
        tools: ToolSet,
        options: LLMOptions,
        client: httpx.AsyncClient,
    ) -> GenerationResult: ...

    @abc.abstractmethod
    def stream(
        self,
        instructions: str,
        messages: list[Message],
        tools: ToolSet,
        options: LLMOptions,
        client: httpx.AsyncClient,
    ) -> AsyncGenerator[StreamPart, None]: ...
