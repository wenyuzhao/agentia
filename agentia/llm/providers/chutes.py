from ._openai_api import OpenAIAPIProvider
import os

# Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models


class Chutes(OpenAIAPIProvider):
    def __init__(
        self, model: str, api_key: str | None = None, base_url: str | None = None
    ):
        api_key = (
            api_key
            or os.environ.get("CHUTES_API_KEY")
            or os.environ.get("CHUTES_API_TOKEN")
        )
        if not api_key:
            raise ValueError(
                "CHUTES_API_KEY or CHUTES_API_TOKEN environment variable not set"
            )
        base_url = "https://llm.chutes.ai/v1"
        super().__init__(
            provider="chutes", model=model, api_key=api_key, base_url=base_url
        )
