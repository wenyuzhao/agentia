from typing import Any, override
from ._openai_api import OpenAIAPIProvider
import os
import httpx

# Model list: https://ollama.com/library

_context_length_cache: dict[str, int] = {}


class Ollama(OpenAIAPIProvider):
    def __init__(
        self, model: str, api_key: str | None = None, base_url: str | None = None
    ):
        base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL",
            os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        )
        api_key = api_key or os.environ.get("OLLAMA_API_KEY", "dummy")
        super().__init__(name="ollama", model=model, api_key=api_key, base_url=base_url)

    @override
    def get_reasoning_args(
        self,
        enabled: bool | None,
        effort: str | None,
        exclude: bool | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        if not enabled:
            return {}
        return {
            "think": enabled,
        }

    @override
    async def _fetch_context_length(self) -> int:
        if self.model in _context_length_cache:
            return _context_length_cache[self.model]
        # Ollama's base_url ends with /v1; the show endpoint is at the Ollama API root
        assert self.base_url is not None
        ollama_base = self.base_url.rstrip("/").removesuffix("/v1")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{ollama_base}/api/show",
                json={"name": self.model},
            )
            resp.raise_for_status()
            data = resp.json()
            model_info = data.get("model_info", {})
            for key, value in model_info.items():
                if "context_length" in key:
                    _context_length_cache[self.model] = int(value)
                    return _context_length_cache[self.model]
            raise ValueError(
                f"Context length not available for model '{self.model}' from Ollama API"
            )
