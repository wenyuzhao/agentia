import os
from agentia.history import History
from . import ModelOptions, LLMBackend
from ..tools import ToolRegistry
from .openai import OpenAIBackend


class DeepSeekBackend(OpenAIBackend):
    def __init__(
        self,
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
        super().__init__(model, tools, options, history, api_key, base_url)
