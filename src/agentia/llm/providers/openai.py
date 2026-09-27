from typing import override
from ._openai_api import OpenAIAPIProvider
from .openrouter import OpenRouter as _OpenRouter
import os

# Model list: https://developers.openai.com/api/docs/models


class OpenAI(OpenAIAPIProvider):
    def __init__(
        self, model: str, api_key: str | None = None, base_url: str | None = None
    ):
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        super().__init__(name="openai", model=model, api_key=api_key, base_url=base_url)

    @override
    async def _fetch_context_length(self) -> int:
        # OpenAI's models endpoint doesn't include context length,
        # so we query OpenRouter's model catalog as a fallback.
        return await _OpenRouter.fetch_context_length(f"openai/{self.model}")
