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


class MessageStream:
    type: Literal["message.stream"] = "message.stream"
    reasoning: Optional["ReasoningMessageStream"] = None

    def __aiter__(self) -> AsyncIterator[str]:
        raise NotImplementedError()

    async def wait_for_completion(self) -> AssistantMessage:
        raise NotImplementedError()


class ReasoningMessageStream:
    type: Literal["message.stream.reasoning"] = "message.stream.reasoning"

    def __aiter__(self) -> AsyncIterator[str]:
        raise NotImplementedError()

    async def wait_for_completion(self) -> str:
        raise NotImplementedError()


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

    async def __await_impl(self) -> str:
        last_message = ""
        async for msg in self:
            if isinstance(msg, Message):
                assert isinstance(msg.content, str)
                last_message = msg.content
            if isinstance(msg, MessageStream):
                last_message = ""
                async for delta in msg:
                    last_message += delta
        return last_message

    def __await__(self):
        return self.__await_impl().__await__()


__all__ = ["ChatCompletion", "MessageStream", "ReasoningMessageStream"]
