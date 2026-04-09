from typing import override
from ._openai_api import OpenAIAPIProvider
import os
import httpx

# Model list: https://openrouter.ai/models

_context_length_cache: dict[str, int] = {}


class OpenRouter(OpenAIAPIProvider):
    def __init__(self, model: str, api_key: str | None = None):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        base_url = "https://openrouter.ai/api/v1"
        super().__init__(
            name="openrouter", model=model, api_key=api_key, base_url=base_url
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

    @staticmethod
    async def fetch_context_length(model: str) -> int:
        """Fetch context length from OpenRouter's model catalog.

        Caches all models from the API response. Can be used by other providers
        that don't expose context length in their own APIs.
        """
        if model in _context_length_cache:
            return _context_length_cache[model]
        async with httpx.AsyncClient() as client:
            resp = await client.get("https://openrouter.ai/api/v1/models")
            resp.raise_for_status()
            for m in resp.json().get("data", []):
                mid = m.get("id")
                ctx = m.get("context_length")
                if mid and ctx is not None:
                    _context_length_cache[mid] = int(ctx)
            if model in _context_length_cache:
                return _context_length_cache[model]
            raise ValueError(
                f"Context length not available for model '{model}' from OpenRouter API"
            )

    @override
    async def _fetch_context_length(self) -> int:
        return await self.fetch_context_length(self.model)
