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


from agentia.message import AssistantMessage, Event, Message
from agentia.utils.session import SessionLock

if TYPE_CHECKING:
    from agentia.agent import Agent


class MessageStream(abc.ABC):
    type: Literal["message.stream"] = "message.stream"
    reasoning: Optional["ReasoningMessageStream"] = None

    def __aiter__(self) -> AsyncIterator[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    async def _wait_for_completion(self) -> AssistantMessage:
        raise NotImplementedError()

    def __await__(self):
        return self._wait_for_completion().__await__()


class ReasoningMessageStream(abc.ABC):
    type: Literal["message.stream.reasoning"] = "message.stream.reasoning"

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


class ChatCompletion(Generic[M]):
    def __init__(
        self,
        agent: "Agent",
        agen: AsyncGenerator[M, None],
    ) -> None:
        super().__init__()
        self.__agen = agen
        self.__agent = agent
        self.__lock: SessionLock = SessionLock(agent)

    async def __save_history(self):
        if not self.__agent.persist:
            return
        await self.__agent.save()

    async def __end_of_stream(self, error: bool):
        if not error:
            await self.__save_history()
        self.__lock.unlock()

    async def __anext__(self) -> M:
        await self.__agent.init()
        try:
            return await self.__agen.__anext__()
        except StopAsyncIteration as e:
            await self.__end_of_stream(False)
            raise e
        except BaseException as e:
            await self.__end_of_stream(True)
            raise e

    def __aiter__(self):
        return self

    async def __await_impl(self) -> AssistantMessage:
        last_message: Message | None = None
        async for msg in self:
            if isinstance(msg, Message):
                last_message = msg
            if isinstance(msg, MessageStream):
                last_message = await msg
        assert last_message is not None
        return last_message

    def __await__(self):
        return self.__await_impl().__await__()


__all__ = ["ChatCompletion", "MessageStream", "ReasoningMessageStream"]
