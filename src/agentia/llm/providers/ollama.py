import os
from typing import Any, override

import httpx

from ._openai_api import OpenAIAPIProvider

# Model list: https://ollama.com/library

_context_length_cache: dict[str, int] = {}


class Ollama(OpenAIAPIProvider):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
        headers: dict[str, str] | None = None,
    ):
        base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL",
            os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        )
        api_key = api_key or os.environ.get("OLLAMA_API_KEY", "dummy")
        super().__init__(
            name="ollama",
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            headers=headers,
        )

    @override
    def get_reasoning_args(
        self,
        enabled: bool | None,
        effort: str | None,
        exclude: bool | None,
        max_tokens: int | None,
    ) -> dict[str, Any] | None:
        if not enabled:
            return None
        return {"think": enabled}

    @override
    async def _fetch_context_length(self) -> int:
        if self.model in _context_length_cache:
            return _context_length_cache[self.model]
        # Ollama's base_url ends with /v1; the show endpoint is at the Ollama API root
        ollama_base = self.base_url.removesuffix("/v1")
        http = self.client or httpx.AsyncClient()
        try:
            resp = await http.post(f"{ollama_base}/api/show", json={"name": self.model})
            resp.raise_for_status()
            for key, value in resp.json().get("model_info", {}).items():
                if "context_length" in key:
                    _context_length_cache[self.model] = int(value)
                    return _context_length_cache[self.model]
        finally:
            if http is not self.client:
                await http.aclose()
        raise ValueError(
            f"Context length not available for model '{self.model}' from Ollama API"
        )
