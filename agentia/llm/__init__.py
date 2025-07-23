import abc
from dataclasses import dataclass
import inspect
import os
from typing import Any, AsyncGenerator, Literal, Optional, Sequence, overload
from openai.lib._parsing._completions import type_to_response_format_param  # type: ignore
from openai.lib._pydantic import _ensure_strict_json_schema
from pydantic import BaseModel, TypeAdapter

from ..tools import ToolRegistry
from ..message import AssistantMessage, Message, Event, is_event, is_message
from ..run import Run, MessageStream
from ..history import History

from dataclasses import dataclass


@dataclass
class ModelOptions:
    frequency_penalty: float | None = None
    # logit_bias: Optional[dict[str, int]] = None
    # max_tokens: Optional[int] = None
    # n: Optional[int] = None
    presence_penalty: float | None = None
    # stop: Optional[str] | List[str] = None
    temperature: float | None = None
    # repetition_penalty: Optional[float] = None
    # top_p: Optional[float] = None
    # timeout: Optional[float] = None

    def as_kwargs(self) -> dict[str, Any]:
        args = {
            "frequency_penalty": self.frequency_penalty,
            # "logit_bias": self.logit_bias,
            # "max_tokens": self.max_tokens,
            # "n": self.n,
            "presence_penalty": self.presence_penalty,
            # "stop": self.stop,
            "temperature": self.temperature,
            # "repetition_penalty": self.repetition_penalty,
            # "top_p": self.top_p,
            # "timeout": self.timeout,
        }
        return {k: v for k, v in args.items() if v is not None}


class LLMBackend:
    def support_tools(self) -> bool:
        return True

    def __init__(
        self,
        *,
        name: str,
        model: str,
        tools: ToolRegistry,
        options: ModelOptions,
        history: History,
    ):
        self.name = name
        self.options = options or ModelOptions()
        self.model = model
        self.tools = tools
        self.history = history
        self.log = tools._agent.log

    @abc.abstractmethod
    def get_api_key(self) -> str: ...

    @classmethod
    def get_default_model(cls) -> str:
        raise NotImplementedError()

    @overload
    def run(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False] = False,
        events: Literal[False] = False,
        response_format: Any | None,
    ) -> Run[AssistantMessage]: ...

    @overload
    def run(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        events: Literal[False] = False,
        response_format: Any | None,
    ) -> Run[MessageStream]: ...

    @overload
    def run(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False] = False,
        events: Literal[True],
        response_format: Any | None,
    ) -> Run[AssistantMessage | Event]: ...

    @overload
    def run(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        events: Literal[True],
        response_format: Any | None,
    ) -> Run[MessageStream | Event]: ...

    def run(
        self,
        messages: Sequence[Message],
        *,
        stream: bool = False,
        events: bool = False,
        response_format: Any | None,
    ) -> (
        Run[MessageStream]
        | Run[AssistantMessage]
        | Run[MessageStream | Event]
        | Run[AssistantMessage | Event]
    ):
        a = self.tools._agent
        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            response_format = type_to_response_format_param(response_format)  # type: ignore
        if response_format is not None and not isinstance(response_format, dict):
            ta = TypeAdapter(response_format)
            schema = ta.json_schema()
            response_format = _ensure_strict_json_schema(schema, path=(), root=schema)
        if stream and events:
            return Run(
                a,
                self._run(
                    messages, stream=True, events=True, response_format=response_format
                ),  # type: ignore
            )
        elif stream:
            return Run(
                a, self._run(messages, stream=True, events=False, response_format=response_format)  # type: ignore
            )
        elif events:
            return Run(
                a, self._run(messages, stream=False, events=True, response_format=response_format)  # type: ignore
            )
        else:
            return Run(
                self.tools._agent, self._run(messages, stream=False, events=False, response_format=response_format)  # type: ignore
            )

    @overload
    async def _chat_completion_request(
        self,
        messages: list[Message],
        stream: Literal[False],
        response_format: Any | None,
    ) -> AssistantMessage: ...

    @overload
    async def _chat_completion_request(
        self,
        messages: Sequence[Message],
        stream: Literal[True],
        response_format: Any | None,
    ) -> MessageStream: ...

    async def _chat_completion_request(
        self,
        messages: Sequence[Message],
        stream: bool,
        response_format: Any | None,
    ) -> AssistantMessage | MessageStream:
        raise NotImplementedError()

    @overload
    async def _run(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False],
        events: Literal[False],
        response_format: Any | None,
    ) -> AsyncGenerator[AssistantMessage, None]: ...

    @overload
    async def _run(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        events: Literal[False],
        response_format: Any | None,
    ) -> AsyncGenerator[MessageStream, None]: ...

    @overload
    async def _run(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False],
        events: Literal[True],
        response_format: Any | None,
    ) -> AsyncGenerator[AssistantMessage | Event, None]: ...

    @overload
    async def _run(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        events: Literal[True],
        response_format: Any | None,
    ) -> AsyncGenerator[MessageStream | Event, None]: ...

    async def _run(
        self,
        messages: Sequence[Message],
        *,
        stream: bool,
        events: bool,
        response_format: Any | None,
    ) -> (
        AsyncGenerator[MessageStream, None]
        | AsyncGenerator[AssistantMessage, None]
        | AsyncGenerator[MessageStream | Event, None]
        | AsyncGenerator[AssistantMessage | Event, None]
    ):
        for m in messages:
            self.log.debug(m)
            self.history.add(m)
        # First completion request
        message: AssistantMessage
        trimmed_history = self.history.get_for_inference()
        if stream:
            s = await self._chat_completion_request(
                trimmed_history, stream=True, response_format=response_format
            )
            if await s._ensure_non_empty():
                yield s
            message = await s
        else:
            message = await self._chat_completion_request(
                trimmed_history, stream=False, response_format=response_format
            )
            if message.content or message.reasoning:
                yield message
        self.history.add(message)
        self.log.debug(message)
        # Run tools and submit results until convergence
        while len(message.tool_calls) > 0:
            # Run tools
            count = 0
            async for event in self.tools.call_tools(message.tool_calls):
                self.log.debug(event)
                if is_message(event):
                    self.history.add(event)
                    count += 1
                else:
                    assert is_event(event), "Event must be a Event object"
                    if events:
                        self.history.add(event)
                        yield event
            trimmed_history = self.history.get_for_inference()
            # Submit results
            message: AssistantMessage
            if stream:
                r = await self._chat_completion_request(
                    trimmed_history, stream=True, response_format=response_format
                )
                if await r._ensure_non_empty():
                    yield r
                message = await r
            else:
                message = await self._chat_completion_request(
                    trimmed_history, stream=False, response_format=response_format
                )
                if message.content or message.reasoning:
                    yield message
            self.history.add(message)
            self.log.debug(message)


def get_default_provider() -> str:
    if p := os.environ.get("AGENTIA_LLM_BACKEND"):
        # The environment variable AGENTIA_LLM_BACKEND is set. use it.
        provider = p.strip().lower()
    elif "OPENROUTER_API_KEY" in os.environ:
        provider = "openrouter"
    elif "DEEPSEEK_API_KEY" in os.environ:
        provider = "deepseek"
    else:
        provider = "openai"
    if provider not in [
        "openai",
        "openrouter",
        "deepseek",
    ]:
        raise ValueError(f"Unknown provider: {provider}")
    return provider


def get_default_model() -> str:
    if model := os.environ.get("AGENTIA_MODEL"):
        return model
    match get_default_provider():
        case "openai":
            from .openai import OpenAIBackend

            return OpenAIBackend.get_default_model()
        case "openrouter":
            from .openrouter import OpenRouterBackend

            return OpenRouterBackend.get_default_model()
        case "deepseek":
            from .deepseek import DeepSeekBackend

            return DeepSeekBackend.get_default_model()
        case p:
            raise ValueError(f"Unknown provider: {p}")


def create_llm_backend(
    *,
    model: str,
    options: Optional["ModelOptions"],
    api_key: str | None,
    tools: ToolRegistry,
    history: History,
) -> LLMBackend:
    """
    :param model: The model name to use.
        * `openai/gpt-3.5-turbo` - by default this will use openrouter backend, if the environment variable `AGENTIA_LLM_BACKEND` is not set.
        * `[openai] gpt-3.5-turbo` - this will use openai backend
    """
    model = model.strip()
    if model.startswith("[") and "]" in model:
        # The model string has a [provider] prefix. parse it.
        provider = model.split("]", 1)[0][1:].strip().lower()
        model = model.split("]", 1)[1].strip()
    else:
        provider = get_default_provider()
    assert provider in [
        "openai",
        "openrouter",
        "deepseek",
    ], f"Unknown provider: {provider}"
    if provider == "openai":
        from .openai import OpenAIBackend

        return OpenAIBackend(
            model=model,
            tools=tools,
            options=options or ModelOptions(),
            history=history,
            api_key=api_key,
        )
    elif provider == "deepseek":
        from .deepseek import DeepSeekBackend

        return DeepSeekBackend(
            model=model,
            tools=tools,
            options=options or ModelOptions(),
            history=history,
            api_key=api_key,
        )
    else:
        from .openrouter import OpenRouterBackend

        return OpenRouterBackend(
            model=model,
            tools=tools,
            options=options or ModelOptions(),
            history=history,
            api_key=api_key,
        )
