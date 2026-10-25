import os
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import AnyUrl, BaseModel, ConfigDict

from agentia import models

if TYPE_CHECKING:
    from agentia.llm.providers import Provider


class ReasoningOptions(BaseModel):
    enabled: bool | None = None
    effort: str | None = None
    exclude: bool | None = None
    max_tokens: int | None = None


class LLMOptions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_output_tokens: int | None = None
    temperature: float | None = None
    stop_sequences: Sequence[str] | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    tool_choice: models.ToolChoice | None = None
    extra_body: dict[str, Any] | None = None
    response_format: models.ResponseFormat | type[BaseModel] | None = None
    parallel_tool_calls: bool | None = True
    reasoning: ReasoningOptions | None = None

    max_steps: int | None = None
    """Stop a run (finish reason `max-steps`) after this many model calls."""

    max_retries: int | None = None
    """Retries per model on 408/429/5xx and network errors (default 3). A response is
    never retried once output has streamed, so nothing is shown twice."""

    fallback_models: Sequence[str] | None = None
    """Models to try, in order, when the primary model keeps failing. Not used for
    authentication or billing errors (401/402/403)."""

    timeout: float | None = None
    """Read timeout in seconds for one streamed response (default 600)."""

    cache_control: bool | None = None
    """Send explicit prompt-cache breakpoints (after the static instructions and after
    the summary). `None` lets the provider decide (OpenRouter: Anthropic and Gemini)."""


def get_provider(
    selector: str,
    *,
    client: httpx.AsyncClient | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
) -> "Provider":
    """Build a provider from a `[provider:][owner/]model[:think]` selector."""
    if re.match(r"^[\w-]+:", selector) is None:
        # Default to openrouter if no provider specified
        default_provider = os.environ.get("AGENTIA_DEFAULT_PROVIDER", "openrouter")
        selector = f"{default_provider.strip().lower()}:{selector}"
    uri = AnyUrl(selector)
    for c in ["username", "password", "host", "port", "query", "fragment"]:
        if getattr(uri, c) is not None:
            raise ValueError(f"Invalid LLM selector: {selector}")
    provider, model = uri.scheme, (uri.path or "").strip("/")
    if not provider or not model:
        raise ValueError(f"Invalid LLM selector: {selector}")
    kw: dict[str, Any] = {
        "client": client,
        "api_key": api_key,
        "base_url": base_url,
        "headers": headers,
    }

    match provider:
        case "openai":
            from .providers.openai import OpenAI

            return OpenAI(model=model, **kw)
        case "vercel":
            from .providers.vercel import Vercel

            return Vercel(model=model, **kw)
        case "cloudflare":
            from .providers.cloudflare import Cloudflare

            return Cloudflare(model=model, **kw)
        case "openrouter":
            from .providers.openrouter import OpenRouter

            return OpenRouter(model=model, **kw)
        case "ollama":
            from .providers.ollama import Ollama

            return Ollama(model=model, **kw)
        case _:
            raise ValueError(f"Unknown provider: {provider}")
