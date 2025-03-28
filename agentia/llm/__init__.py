from dataclasses import dataclass
from logging import Logger
from typing import Any, AsyncGenerator, Literal, Sequence, overload

from ..tools import ToolRegistry
from ..message import AssistantMessage, Message, MessageStream
from ..agent import ChatCompletion
from ..history import History

from dataclasses import dataclass
from .. import MSG_LOGGER


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
        model: str,
        tools: ToolRegistry,
        options: ModelOptions,
        history: History,
    ):
        self.options = options or ModelOptions()
        self.model = model
        self.tools = tools
        self.history = history
        self.log = tools._agent.log

    @overload
    def chat_completion(
        self, messages: Sequence[Message], stream: Literal[False] = False
    ) -> ChatCompletion[AssistantMessage]: ...

    @overload
    def chat_completion(
        self, messages: Sequence[Message], stream: Literal[True]
    ) -> ChatCompletion[MessageStream]: ...

    def chat_completion(
        self, messages: Sequence[Message], stream: bool = False
    ) -> ChatCompletion[MessageStream] | ChatCompletion[AssistantMessage]:
        if stream:
            return ChatCompletion(
                self.tools._agent, self._chat_completion(messages, stream=True)
            )
        else:
            return ChatCompletion(
                self.tools._agent, self._chat_completion(messages, stream=False)
            )

    async def _on_new_chat_message(self, msg: Message):
        await self.tools.on_new_chat_message(msg)

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
    ) -> Message | MessageStream:
        raise NotImplementedError

    @overload
    async def _chat_completion(
        self, messages: Sequence[Message], stream: Literal[False]
    ) -> AsyncGenerator[AssistantMessage, None]: ...

    @overload
    async def _chat_completion(
        self, messages: Sequence[Message], stream: Literal[True]
    ) -> AsyncGenerator[MessageStream, None]: ...

    async def _chat_completion(
        self, messages: Sequence[Message], stream: bool
    ) -> AsyncGenerator[AssistantMessage | MessageStream, None]:
        for m in messages:
            self.log.info(f"{m}")
            self.history.add(m)
            await self._on_new_chat_message(m)
        # First completion request
        message: AssistantMessage
        trimmed_history = self.history.get_for_inference(keep_last=len(messages))
        if stream:
            s = await self._chat_completion_request(trimmed_history, stream=True)
            yield s
            message = await s.wait_for_completion()
        else:
            message = await self._chat_completion_request(trimmed_history, stream=False)
            if message.content is not None:
                yield message
        self.history.add(message)
        self.log.info(f"{message}")
        await self._on_new_chat_message(message)
        # Run tools and submit results until convergence
        while len(message.tool_calls) > 0:
            # Run tools
            count = 0
            async for event in self.tools.call_tools(message.tool_calls):
                if isinstance(event, Message):
                    self.history.add(event)
                    count += 1
                else:
                    yield event
            trimmed_history = self.history.get_for_inference(keep_last=count + 1)
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
            self.log.info(f"{message}")
            await self._on_new_chat_message(message)
