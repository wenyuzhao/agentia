import os
import re
from typing import TYPE_CHECKING, Sequence
from openai import BaseModel
from pydantic import AnyUrl
from agentia import spec

if TYPE_CHECKING:
    from agentia.llm.providers import Provider


class ReasoningOptions(BaseModel):
    enabled: bool | None = None
    effort: str | None = None
    exclude: bool | None = None
    max_tokens: int | None = None


class LLMOptions(BaseModel):
    max_output_tokens: int | None = None
    temperature: float | None = None
    stop_sequences: Sequence[str] | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    tool_choice: spec.ToolChoice | None = None
    provider_options: spec.ProviderOptions | None = None
    response_format: spec.ResponseFormat | type[BaseModel] | None = None
    parallel_tool_calls: bool | None = True
    reasoning: ReasoningOptions | None = None


def get_provider(selector: str) -> "Provider":
    # Parse selector
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

    # Instantiate provider
    match provider:
        case "openai":
            from .providers.openai import OpenAI

            return OpenAI(model=model)
        case "vercel":
            from .providers.vercel import Vercel

            return Vercel(model=model)
        case "cloudflare":
            from .providers.cloudflare import Cloudflare

            return Cloudflare(model=model)
        case "openrouter":
            from .providers.openrouter import OpenRouter

            return OpenRouter(model=model)
        case "qwen":
            from .providers.qwen import Qwen

            return Qwen(model=model)
        case "chutes":
            from .providers.chutes import Chutes

            return Chutes(model=model)
        case "fireworks":
            from .providers.fireworks import Fireworks

            return Fireworks(model=model)
        case "ollama":
            from .providers.ollama import Ollama

            return Ollama(model=model)
        case "gemini-live":
            from .providers.gemini_live import GeminiLive

            return GeminiLive(model=model)
        case _:
            raise ValueError(f"Unknown provider: {provider}")
