from ._openai_api import OpenAIAPIProvider
import os


class Gateway(OpenAIAPIProvider):
    def __init__(
        self, model: str, api_key: str | None = None, base_url: str | None = None
    ):
        api_key = api_key or os.environ.get("AI_GATEWAY_API_KEY")
        if not api_key:
            raise ValueError("AI_GATEWAY_API_KEY environment variable not set")
        base_url = "https://ai-gateway.vercel.sh/v1"
        super().__init__(
            provider="gateway", model=model, api_key=api_key, base_url=base_url
        )
