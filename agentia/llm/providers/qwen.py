from ._openai_api import OpenAIAPIProvider
import os

# Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models


class Qwen(OpenAIAPIProvider):
    def __init__(
        self, model: str, api_key: str | None = None, base_url: str | None = None
    ):
        api_key = (
            api_key
            or os.environ.get("QWEN_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "QWEN_API_KEY or DASHSCOPE_API_KEY environment variable not set"
            )
        base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        super().__init__(
            provider="qwen", model=model, api_key=api_key, base_url=base_url
        )
