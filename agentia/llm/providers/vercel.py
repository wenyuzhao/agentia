from ._openai_api import OpenAIAPIProvider
import os

# Model list: https://vercel.com/ai-gateway/models


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
