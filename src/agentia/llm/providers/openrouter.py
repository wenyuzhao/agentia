import os
import re
from typing import Any, override

import httpx

from ._openai_api import OpenAIAPIProvider

# Model list: https://openrouter.ai/models

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouter(OpenAIAPIProvider):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
        headers: dict[str, str] | None = None,
    ):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        base_url = base_url or os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL
        super().__init__(
            name="openrouter",
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            headers=headers,
        )
        m = os.getenv("AGENTIA_OPENROUTER_MODALITIES", None)
        if m:
            self.extra_body["modalities"] = [
                x.strip().lower() for x in m.split(",") if x.strip()
            ]
        t = os.getenv("AGENTIA_OPENROUTER_TRANSFORMS", None)
        if t and t != "0" and t.lower() != "false":
            self.extra_body["transforms"] = ["middle-out"]
        if p := os.getenv("OPENROUTER_PROVIDERS", None):
            self.extra_body["provider"] = {
                "order": [x.strip().lower() for x in p.split(",") if x.strip()],
            }

    @override
    def default_cache_control(self) -> bool:
        # Anthropic and Gemini only cache at explicit breakpoints; others cache
        # automatically.
        return bool(re.match(r"^(anthropic/|google/gemini)", self.model))

    @override
    def get_reasoning_args(
        self,
        enabled: bool | None,
        effort: str | None,
        exclude: bool | None,
        max_tokens: int | None,
    ) -> dict[str, Any] | None:
        """Extend the Responses `reasoning` object with OpenRouter's own fields.

        `enabled` is required for models where reasoning is opt-in (e.g. Anthropic)
        and no effort is given, and `effort` and `max_tokens` are mutually exclusive.
        """
        if not enabled:
            return None
        reasoning: dict[str, Any] = {"enabled": True}
        if effort is not None:
            reasoning["effort"] = effort
        elif max_tokens is not None:
            reasoning["max_tokens"] = max_tokens
        if exclude is not None:
            reasoning["exclude"] = exclude
        if not exclude:
            reasoning["summary"] = "auto"
        return reasoning

    @staticmethod
    async def fetch_context_length(
        model: str, client: httpx.AsyncClient | None = None
    ) -> int:
        """Context length from OpenRouter's public model catalog.

        Other providers whose APIs don't expose context lengths use this as a fallback.
        """
        p = OpenRouter.__new__(OpenRouter)
        p.client, p.api_key, p.base_url = client, "", DEFAULT_BASE_URL
        return await p._context_length_from(
            f"{DEFAULT_BASE_URL}/models", model, auth=False
        )
