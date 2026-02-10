from ._openai_api import OpenAIAPIProvider
import os


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
        super().__init__(
            provider="openai", model=model, api_key=api_key, base_url=base_url
        )
