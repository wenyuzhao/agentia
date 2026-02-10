import abc
import re
from dataclasses import dataclass
from typing import AsyncGenerator, Sequence
import httpx
from agentia.llm import LLMOptions
from agentia.spec.chat import Message
from agentia.tools.tools import ToolSet
import agentia.spec as spec
from agentia.spec.stream import StreamPart


@dataclass
class GenerationResult:
    message: spec.AssistantMessage
    finish_reason: spec.FinishReason
    usage: spec.Usage
    provider_metadata: spec.ProviderMetadata | None


class Provider(abc.ABC):
    provider: str
    model: str
    supported_urls: dict[str, Sequence[re.Pattern | str]]
    """
    Supported URL patterns by media type for the provider.

    The keys are media type patterns or full media types (e.g. `*\\/*` for everything, `audio/*`, `video/*`, or `application/pdf`).
    and the values are arrays of regular expressions that match the URL paths.

    The matching should be against lower-case URLs.

    Matched URLs are supported natively by the model and are not downloaded.
    """

    def __init__(self, provider: str, model: str):
        self.provider = provider
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
