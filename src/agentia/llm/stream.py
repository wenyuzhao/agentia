from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import TYPE_CHECKING, Literal

from agentia.llm.completion import RunBase, async_gen_to_sync, last_assistant_message
from agentia.models import *

if TYPE_CHECKING:
    from agentia.agent import Agent


class TextStream:
    def __init__(self, gen: AsyncGenerator[str, None]):
        async def _gen():
            async for part in gen:
                self.__all += part
                yield part

        self.__all = ""
        self.__gen = _gen()

    def __aiter__(self) -> AsyncIterator[str]:
        return self.__gen

    async def __wait_for_completion(self) -> str:
        async for _ in self.__gen:
            pass
        return self.__all

    def __await__(self):
        return self.__wait_for_completion().__await__()

    def __iter__(self):
        return async_gen_to_sync(self.__gen)


class ReasoningStream(TextStream):
    type: Literal["reasoning-stream"] = "reasoning-stream"


class MessageStream(TextStream):
    type: Literal["message-stream"] = "message-stream"


async def _text_stream(
    gen: AsyncGenerator[StreamEvent, None], first: str
) -> AsyncGenerator[str, None]:
    """Yield output-text deltas until the current content part ends."""
    yield first
    async for event in gen:
        match event.type:
            case "response.output_text.delta":
                yield event.delta
            case "response.output_text.done" | "response.content_part.done":
                break


async def _reasoning_stream(
    gen: AsyncGenerator[StreamEvent, None], first: str
) -> AsyncGenerator[str, None]:
    """Yield reasoning deltas until the current reasoning item ends."""
    yield first
    async for event in gen:
        match event.type:
            case (
                "response.reasoning_text.delta"
                | "response.reasoning_summary_text.delta"
            ):
                yield event.delta
            case (
                "response.reasoning_text.done"
                | "response.reasoning_summary_text.done"
                | "response.output_item.done"
            ):
                break


class RunStream(RunBase):
    """A streaming `Agent.run()`, yielding text/reasoning streams and tool items."""

    def __init__(
        self,
        make: Callable[[RunBase], AsyncGenerator[StreamEvent, None]],
        agent: "Agent",
    ):
        super().__init__(agent)
        gen = make(self)

        async def __gen() -> AsyncGenerator[
            ReasoningStream | MessageStream | FunctionCall | FunctionCallOutput, None
        ]:
            n = 0

            def committed() -> list[FunctionCall | FunctionCallOutput]:
                # Calls and results as committed to the history (with final call ids).
                nonlocal n
                new = self.new_items[n:]
                n = len(self.new_items)
                return [
                    i for i in new if isinstance(i, (FunctionCall, FunctionCallOutput))
                ]

            async for event in gen:
                for item in committed():
                    yield item
                match event.type:
                    case "response.output_text.delta":
                        yield MessageStream(_text_stream(gen, event.delta))
                    case (
                        "response.reasoning_text.delta"
                        | "response.reasoning_summary_text.delta"
                    ):
                        yield ReasoningStream(_reasoning_stream(gen, event.delta))
            for item in committed():
                yield item

        self.__gen = __gen()

    def __aiter__(
        self,
    ) -> AsyncIterator[
        ReasoningStream | MessageStream | FunctionCall | FunctionCallOutput
    ]:
        return self.__gen

    async def __wait_for_completion(self) -> Message:
        async for _ in self.__gen:
            ...
        return last_assistant_message(self.new_items)

    def __await__(self):
        return self.__wait_for_completion().__await__()

    def __iter__(self):
        return async_gen_to_sync(self.__gen)


class RunEvents(RunBase):
    """A streaming `Agent.run()`, yielding raw Responses API stream events."""

    def __init__(
        self,
        make: Callable[[RunBase], AsyncGenerator[StreamEvent, None]],
        agent: "Agent",
    ):
        super().__init__(agent)
        self.__gen = make(self)

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self.__gen

    async def __wait_for_completion(self) -> Message:
        async for _ in self.__gen:
            ...
        return last_assistant_message(self.new_items)

    def __await__(self):
        return self.__wait_for_completion().__await__()
