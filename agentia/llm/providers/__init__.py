import abc
import re
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Awaitable, Coroutine, Sequence
from pydantic import BaseModel
from agentia.llm import GenerationOptions
import agentia.spec as spec


@dataclass
class ProviderGenerationResult:
    content: Sequence[spec.Content]
    finish_reason: spec.FinishReason
    usage: spec.Usage
    warnings: Sequence[spec.Warning]
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
    async def do_generate(
        self, prompt: spec.Prompt, options: GenerationOptions
    ) -> ProviderGenerationResult: ...

    @abc.abstractmethod
    def do_stream(
        self, prompt: spec.Prompt, options: GenerationOptions
    ) -> AsyncGenerator[spec.StreamPart, None]: ...
