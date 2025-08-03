import abc
from typing import (
    TYPE_CHECKING,
    AsyncGenerator,
    AsyncIterator,
    Generic,
    Literal,
    Optional,
    TypeVar,
)


from agentia.message import AssistantMessage, Event, Message, is_message

if TYPE_CHECKING:
    from agentia.agent import Agent


class MessageStream(abc.ABC):
    type: Literal["message.stream"] = "message.stream"
    reasoning: Optional["ReasoningMessageStream"] = None

    def __aiter__(self) -> AsyncIterator[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    async def _ensure_non_empty(self) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    async def _wait_for_completion(self) -> AssistantMessage:
        raise NotImplementedError()

    def __await__(self):
        return self._wait_for_completion().__await__()


class ReasoningMessageStream(abc.ABC):
    type: Literal["message.stream.reasoning"] = "message.stream.reasoning"

    @abc.abstractmethod
    async def _ensure_non_empty(self) -> bool:
        raise NotImplementedError()

    def __aiter__(self) -> AsyncIterator[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    async def _wait_for_completion(self) -> str:
        raise NotImplementedError()

    def __await__(self):
        return self._wait_for_completion().__await__()


M = TypeVar(
    "M",
    AssistantMessage,
    MessageStream,
    AssistantMessage | Event,
    MessageStream | Event,
)


class Run(Generic[M]):
    def __init__(
        self,
        agent: "Agent",
        agen: AsyncGenerator[M, None],
    ) -> None:
        super().__init__()
        self.__agen = agen
        self.__agent = agent
        self.__last_message_or_stream: Optional[M] = None

    @property
    def agent(self) -> "Agent":
        return self.__agent

    async def __anext__(self) -> M:
        await self.__agent.init()
        item = await self.__agen.__anext__()
        self.__last_message_or_stream = item
        return item

    def __aiter__(self):
        return self

    async def __await_impl(self) -> AssistantMessage:
        async for msg in self:
            ...
        last_message: Message | None = None
        assert self.__last_message_or_stream is not None
        if is_message(self.__last_message_or_stream):
            last_message = self.__last_message_or_stream
        elif isinstance(self.__last_message_or_stream, MessageStream):
            last_message = await self.__last_message_or_stream._wait_for_completion()
        assert last_message is not None
        assert isinstance(
            last_message, AssistantMessage
        ), "Last message must be an AssistantMessage"
        return last_message

    def __await__(self):
        return self.__await_impl().__await__()


__all__ = ["Run", "MessageStream", "ReasoningMessageStream"]
