from ._openai_api import OpenAIAPIProvider
import os


class OpenRouter(OpenAIAPIProvider):
    def __init__(
        self, model: str, api_key: str | None = None, base_url: str | None = None
    ):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        base_url = "https://openrouter.ai/api/v1"
        super().__init__(
            provider="openrouter", model=model, api_key=api_key, base_url=base_url
        )
