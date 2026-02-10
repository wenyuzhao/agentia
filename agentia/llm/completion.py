import asyncio
from typing import Any, AsyncGenerator, Generator, TYPE_CHECKING
from agentia.spec.base import FinishReason, Usage
from agentia.spec.chat import AssistantMessage, Message, NonSystemMessage, ToolMessage

if TYPE_CHECKING:
    from agentia.agent import Agent


class Listeners:
    def __init__(self):
        self.__listeners = []

    def on(self, listener):
        self.__listeners.append(listener)

    def off(self, listener):
        self.__listeners.remove(listener)

    def emit(self, *args, **kwargs):
        for listener in self.__listeners:
            listener(*args, **kwargs)


class ChatCompletion:
    def __init__(
        self, gen: AsyncGenerator[AssistantMessage | ToolMessage, None], agent: "Agent"
    ):
        async def __gen() -> AsyncGenerator[AssistantMessage | ToolMessage, None]:
            async for msg in gen:
                yield msg
            self._on_finish()

        self.__gen = __gen()
        self.usage = Usage()
        self.finish_reason: FinishReason | None = None
        self.new_messages: list[NonSystemMessage] = []
        self.on_finish = Listeners()
        self.agent = agent

    def _on_finish(self):
        self.on_finish.emit()
        self.agent.emit("finish")

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
