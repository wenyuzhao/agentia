import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Generator
from typing import TYPE_CHECKING

from agentia.models.base import FinishReason, Usage
from agentia.models.items import Item, Message
from agentia.models.stream import StreamEvent

if TYPE_CHECKING:
    from agentia.agent import Agent


class RunBase:
    def __init__(self, agent: "Agent"):
        self.usage = Usage()
        self.last_usage = Usage()
        self.finish_reason: FinishReason | None = None
        self.new_items: list[Item] = []
        self.agent = agent

    @property
    def suspended(self) -> bool:
        """The run paused on a suspended tool; continue with `agent.run(None)`."""
        return self.finish_reason == "suspended"


class Run(RunBase):
    """A non-streaming `Agent.run()`.

    Iterating yields each new conversation `Item` as it is committed to the history;
    awaiting returns the last assistant `message` item.
    """

    def __init__(
        self,
        make: "Callable[[RunBase], AsyncGenerator[StreamEvent, None]]",
        agent: "Agent",
    ):
        super().__init__(agent)
        events = make(self)

        async def __gen() -> AsyncGenerator[Item, None]:
            n = 0
            async for _ in events:
                # Items are yielded once committed to the history (with final call ids).
                while n < len(self.new_items):
                    yield self.new_items[n]
                    n += 1
            while n < len(self.new_items):
                yield self.new_items[n]
                n += 1

        self.__gen = __gen()

    def __aiter__(self) -> AsyncIterator[Item]:
        return self.__gen

    async def __wait_for_completion(self) -> Message:
        async for _ in self.__gen:
            ...
        return last_assistant_message(self.new_items)

    def __await__(self):
        return self.__wait_for_completion().__await__()

    def __iter__(self):
        return async_gen_to_sync(self.__gen)


def last_assistant_message(items: list[Item]) -> Message:
    """The final assistant `message` item, or an empty one if the run produced none."""
    for item in reversed(items):
        if isinstance(item, Message) and item.role == "assistant":
            return item
    return Message.assistant([])


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
