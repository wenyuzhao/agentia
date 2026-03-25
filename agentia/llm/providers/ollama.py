from typing import Any, override
from ._openai_api import OpenAIAPIProvider
import os


class Ollama(OpenAIAPIProvider):
    def __init__(
        self, model: str, api_key: str | None = None, base_url: str | None = None
    ):
        base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL",
            os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        )
        api_key = api_key or os.environ.get("OLLAMA_API_KEY", "dummy")
        super().__init__(
            provider="ollama", model=model, api_key=api_key, base_url=base_url
        )

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
