import abc
from dataclasses import dataclass
from logging import Logger
import os
from typing import Any, AsyncGenerator, Literal, Optional, Sequence, overload

from ..tools import ToolRegistry
from ..message import AssistantMessage, Message, Event
from ..chat_completion import ChatCompletion, MessageStream
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

    @abc.abstractmethod
    def get_default_model(self) -> str: ...

    @overload
    def chat_completion(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False] = False,
        events: Literal[False] = False,
        response_format: Any | None,
    ) -> ChatCompletion[AssistantMessage]: ...

    @overload
    def chat_completion(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        events: Literal[False] = False,
        response_format: Any | None,
    ) -> ChatCompletion[MessageStream]: ...

    @overload
    def chat_completion(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False] = False,
        events: Literal[True],
        response_format: Any | None,
    ) -> ChatCompletion[AssistantMessage | Event]: ...

    @overload
    def chat_completion(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        events: Literal[True],
        response_format: Any | None,
    ) -> ChatCompletion[MessageStream | Event]: ...

    def chat_completion(
        self,
        messages: Sequence[Message],
        *,
        stream: bool = False,
        events: bool = False,
        response_format: Any | None,
    ) -> (
        ChatCompletion[MessageStream]
        | ChatCompletion[AssistantMessage]
        | ChatCompletion[MessageStream | Event]
        | ChatCompletion[AssistantMessage | Event]
    ):
        a = self.tools._agent
        if stream and events:
            return ChatCompletion(
                a,
                self._chat_completion(
                    messages, stream=True, events=True, response_format=response_format
                ),  # type: ignore
            )
        elif stream:
            return ChatCompletion(
                a, self._chat_completion(messages, stream=True, events=False, response_format=response_format)  # type: ignore
            )
        elif events:
            return ChatCompletion(
                a, self._chat_completion(messages, stream=False, events=True, response_format=response_format)  # type: ignore
            )
        else:
            return ChatCompletion(
                self.tools._agent, self._chat_completion(messages, stream=False, events=False, response_format=response_format)  # type: ignore
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
    async def _chat_completion(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False],
        events: Literal[False],
        response_format: Any | None,
    ) -> AsyncGenerator[AssistantMessage, None]: ...

    @overload
    async def _chat_completion(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        events: Literal[False],
        response_format: Any | None,
    ) -> AsyncGenerator[MessageStream, None]: ...

    @overload
    async def _chat_completion(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False],
        events: Literal[True],
        response_format: Any | None,
    ) -> AsyncGenerator[AssistantMessage | Event, None]: ...

    @overload
    async def _chat_completion(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        events: Literal[True],
        response_format: Any | None,
    ) -> AsyncGenerator[MessageStream | Event, None]: ...

    async def _chat_completion(
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
            yield s
            message = await s
        else:
            message = await self._chat_completion_request(
                trimmed_history, stream=False, response_format=response_format
            )
            if message.content is not None:
                yield message
        self.history.add(message)
        self.log.debug(message)
        # Run tools and submit results until convergence
        while len(message.tool_calls) > 0:
            # Run tools
            count = 0
            async for event in self.tools.call_tools(message.tool_calls):
                self.log.debug(event)
                if isinstance(event, Message):
                    self.history.add(event)
                    count += 1
                else:
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
                yield r
                message = await r
            else:
                message = await self._chat_completion_request(
                    trimmed_history, stream=False, response_format=response_format
                )
                if message.content is not None:
                    yield message
            self.history.add(message)
            self.log.debug(message)


def create_llm_backend(
    *,
    model: str,
    options: Optional["ModelOptions"],
    api_key: str | None,
    tools: ToolRegistry,
    history: History,
) -> LLMBackend:
    if ":" in model:
        provider = model.split(":")[0]
        model = model.split(":")[1]
    elif "OPENAI_BASE_URL" in os.environ:
        provider = "openai"
    else:
        provider = "openrouter"
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
