from typing import override
from ._openai_api import OpenAIAPIProvider
import os
import httpx

# Model list: https://vercel.com/ai-gateway/models

_context_length_cache: dict[str, int] = {}


class Vercel(OpenAIAPIProvider):
    def __init__(self, model: str, api_key: str | None = None):
        api_key = (
            api_key
            or os.environ.get("VERCEL_AI_GATEWAY_API_KEY")
            or os.environ.get("VERCEL_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "VERCEL_AI_GATEWAY_API_KEY or VERCEL_API_KEY environment variable not set"
            )
        base_url = "https://ai-gateway.vercel.sh/v1"
        super().__init__(name="vercel", model=model, api_key=api_key, base_url=base_url)
        self.extra_body["modalities"] = ["text", "image"]

    @override
    async def _fetch_context_length(self) -> int:
        if self.model in _context_length_cache:
            return _context_length_cache[self.model]
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            resp.raise_for_status()
            for model in resp.json().get("data", []):
                mid = model.get("id")
                ctx = model.get("context_window")
                if mid and ctx is not None:
                    _context_length_cache[mid] = int(ctx)
            if self.model in _context_length_cache:
                return _context_length_cache[self.model]
            raise ValueError(
                f"Context length not available for model '{self.model}' from Vercel API"
            )
