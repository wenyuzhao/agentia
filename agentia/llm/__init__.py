import abc
from dataclasses import dataclass
from logging import Logger
from typing import Any, AsyncGenerator, Literal, Sequence, overload

from ..tools import ToolRegistry
from ..message import AssistantMessage, Message, MessageStream
from ..agent import ChatCompletion, Event
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
    ) -> ChatCompletion[AssistantMessage]: ...

    @overload
    def chat_completion(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        events: Literal[False] = False,
    ) -> ChatCompletion[MessageStream]: ...

    @overload
    def chat_completion(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False] = False,
        events: Literal[True],
    ) -> ChatCompletion[AssistantMessage | Event]: ...

    @overload
    def chat_completion(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        events: Literal[True],
    ) -> ChatCompletion[MessageStream | Event]: ...

    def chat_completion(
        self, messages: Sequence[Message], *, stream: bool = False, events: bool = False
    ) -> (
        ChatCompletion[MessageStream]
        | ChatCompletion[AssistantMessage]
        | ChatCompletion[MessageStream | Event]
        | ChatCompletion[AssistantMessage | Event]
    ):
        a = self.tools._agent
        if stream and events:
            return ChatCompletion(
                a, self._chat_completion(messages, stream=True, events=True)  # type: ignore
            )
        elif stream:
            return ChatCompletion(
                a, self._chat_completion(messages, stream=True, events=False)  # type: ignore
            )
        elif events:
            return ChatCompletion(
                a, self._chat_completion(messages, stream=False, events=True)  # type: ignore
            )
        else:
            return ChatCompletion(
                self.tools._agent, self._chat_completion(messages, stream=False, events=False)  # type: ignore
            )

    @overload
    async def _chat_completion_request(
        self, messages: list[Message], stream: Literal[False]
    ) -> AssistantMessage: ...

    @overload
    async def _chat_completion_request(
        self, messages: Sequence[Message], stream: Literal[True]
    ) -> MessageStream: ...

    async def _chat_completion_request(
        self, messages: Sequence[Message], stream: bool
    ) -> AssistantMessage | MessageStream:
        raise NotImplementedError()

    @overload
    async def _chat_completion(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False],
        events: Literal[False],
    ) -> AsyncGenerator[AssistantMessage, None]: ...

    @overload
    async def _chat_completion(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        events: Literal[False],
    ) -> AsyncGenerator[MessageStream, None]: ...

    @overload
    async def _chat_completion(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False],
        events: Literal[True],
    ) -> AsyncGenerator[AssistantMessage | Event, None]: ...

    @overload
    async def _chat_completion(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        events: Literal[True],
    ) -> AsyncGenerator[MessageStream | Event, None]: ...

    async def _chat_completion(
        self, messages: Sequence[Message], *, stream: bool, events: bool
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
            s = await self._chat_completion_request(trimmed_history, stream=True)
            yield s
            message = await s.wait_for_completion()
        else:
            message = await self._chat_completion_request(trimmed_history, stream=False)
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
                r = await self._chat_completion_request(trimmed_history, stream=True)
                yield r
                message = await r.wait_for_completion()
            else:
                message = await self._chat_completion_request(
                    trimmed_history, stream=False
                )
                if message.content is not None:
                    yield message
            self.history.add(message)
            self.log.debug(message)
