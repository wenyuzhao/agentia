import asyncio
from typing import AsyncGenerator, Generator, TYPE_CHECKING, Any
from agentia.spec.base import FinishReason, Usage
from agentia.spec.chat import AssistantMessage, NonSystemMessage, ToolMessage

if TYPE_CHECKING:
    from agentia.agent import Agent


class ChatCompletion:
    def __init__(
        self, gen: AsyncGenerator[AssistantMessage | ToolMessage, None], agent: "Agent"
    ):
        async def __gen() -> AsyncGenerator[AssistantMessage | ToolMessage, None]:
            async for msg in gen:
                yield msg
            await self._on_finish()

        self.__gen = __gen()
        self.usage = Usage()
        self.finish_reason: FinishReason | None = None
        self.new_messages: list[NonSystemMessage] = []
        self.agent = agent

    async def _on_finish(self):
        await self.agent.events.end_of_turn.emit()

    def _add_new_message(self, msg: NonSystemMessage):
        self.new_messages.append(msg)

    def __aiter__(self) -> AsyncGenerator[AssistantMessage | ToolMessage, None]:
        return self.__gen

    async def __wait_for_completion(self) -> AssistantMessage:
        async for msg in self.__gen:
            ...
        m = self.new_messages[-1]
        assert isinstance(m, AssistantMessage)
        return m

    def __await__(self):
        return self.__wait_for_completion().__await__()

    def __iter__(self):
        return async_gen_to_sync(self.__gen)


def async_gen_to_sync[T](agen: AsyncGenerator[T, None]) -> Generator[T, None, None]:
    try:
        loop = asyncio.get_running_loop()
        close = False
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio._set_running_loop(loop)
        close = True
    try:
        while True:
            try:
                item = loop.run_until_complete(agen.__anext__())
                yield item
            except StopAsyncIteration:
                break
    finally:
        if close:
            loop.close()
            asyncio._set_running_loop(None)
