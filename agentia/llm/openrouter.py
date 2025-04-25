import os
from typing import Any
from agentia.history import History
from . import ModelOptions
from ..tools import ToolRegistry
from .openai import OpenAIBackend
import requests


class OpenRouterBackend(OpenAIBackend):
    def __init__(
        self,
        *,
        model: str,
        tools: ToolRegistry,
        options: ModelOptions,
        history: History,
        api_key: str | None = None,
    ) -> None:
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        base_url = "https://openrouter.ai/api/v1"
        super().__init__(
            name="openrouter",
            model=model,
            tools=tools,
            options=options,
            history=history,
            api_key=api_key,
            base_url=base_url,
        )
        if providers := os.environ.get("OPENROUTER_PROVIDERS"):
            self.extra_body["provider"] = {
                "order": [x.strip() for x in providers.strip().split(",")]
            }
        if v := os.environ.get("OPENROUTER_INCLUDE_REASONING"):
            include_reasoning = v.lower() in ["true", "1", "yes", "y"]
            model_has_reasoning = self.__model_has_reasoning(model)
            if not model_has_reasoning:
                include_reasoning = False
        else:
            model_has_reasoning = self.__model_has_reasoning(model)
            include_reasoning = model_has_reasoning
        if model_has_reasoning:
            reasoning: dict[str, Any] = {
                "exclude": not include_reasoning,
            }
            if effort := os.environ.get("OPENROUTER_REASONING_EFFORT"):
                if effort.lower() in ["high", "medium", "low"]:
                    reasoning["effort"] = effort.lower()
            self.extra_body["reasoning"] = reasoning
        self.has_reasoning = model_has_reasoning and include_reasoning
        self.extra_body["transforms"] = ["middle-out"]

    def get_default_model(self) -> str:
        return "openrouter:openai/gpt-4o-mini"

    def __model_has_reasoning(self, model: str):
        if v := os.environ.get("OPENROUTER_HAS_REASONING"):
            return v.lower() in ["true", "1", "yes", "y"]
        global _REASONING_MODELS
        if model in _REASONING_MODELS:
            return _REASONING_MODELS[model]
        res = requests.get(f"https://openrouter.ai/api/v1/models/{model}/endpoints")
        endpoints = res.json().get("data", {}).get("endpoints", [])
        has_reasoning = False
        if len(endpoints) > 0:
            e = endpoints[0]
            has_reasoning = "include_reasoning" in e.get("supported_parameters", [])
        _REASONING_MODELS[model] = has_reasoning
        return has_reasoning


_REASONING_MODELS: dict[str, bool] = {}
