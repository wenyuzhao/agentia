import asyncio
from typing import AsyncGenerator, Generator, TYPE_CHECKING
from agentia.models.base import FinishReason, Usage
from agentia.models.items import Item, Message

if TYPE_CHECKING:
    from agentia.agent import Agent


class Run:
    """A non-streaming `Agent.run()`.

    Iterating yields each new conversation `Item` as it is produced; awaiting returns
    the last assistant `message` item.
    """

    def __init__(self, gen: AsyncGenerator[Item, None], agent: "Agent"):
        async def __gen() -> AsyncGenerator[Item, None]:
            async for item in gen:
                yield item
            await self._on_finish()

        self.__gen = __gen()
        self.usage = Usage()
        self.last_usage = Usage()
        self.finish_reason: FinishReason | None = None
        self.new_items: list[Item] = []
        self.agent = agent

    async def _on_finish(self):
        if self.last_usage.total_tokens is not None:
            self.agent.history.current_tokens = self.last_usage.total_tokens
        self.agent.history.usage += self.usage
        await self.agent.events.end_of_turn.emit()

    def _add_new_item(self, item: Item):
        self.new_items.append(item)

    def __aiter__(self) -> AsyncGenerator[Item, None]:
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
