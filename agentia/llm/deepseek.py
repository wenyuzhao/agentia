import os
from typing import override
from agentia.history import History
from . import ModelOptions, LLMBackend
from ..tools import ToolRegistry
from .openai import OpenAIBackend


class DeepSeekBackend(OpenAIBackend):
    def __init__(
        self,
        *,
        model: str,
        tools: ToolRegistry,
        options: ModelOptions,
        history: History,
        api_key: str | None = None,
    ) -> None:
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
        base_url = "https://api.deepseek.com"
        super().__init__(
            name="deepseek",
            model=model,
            tools=tools,
            options=options,
            history=history,
            api_key=api_key,
            base_url=base_url,
        )

    @override
    def get_default_model(self) -> str:
        return "deepseek-chat"
