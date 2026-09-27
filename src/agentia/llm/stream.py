from typing import AsyncGenerator, Literal, TYPE_CHECKING

from agentia.llm.completion import async_gen_to_sync, last_assistant_message
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

    def __aiter__(self) -> AsyncGenerator[str, None]:
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


class RunStreamBase:
    def __init__(self, agent: "Agent"):
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


class RunStream(RunStreamBase):
    """A streaming `Agent.run()`, yielding text/reasoning streams and finished items."""

    def __init__(self, gen: AsyncGenerator[StreamEvent, None], agent: "Agent"):
        super().__init__(agent)

        async def __gen() -> AsyncGenerator[
            ReasoningStream | MessageStream | FunctionCall | FunctionCallOutput, None
        ]:
            async for event in gen:
                match event.type:
                    case "response.output_item.done":
                        # Only the model's own function calls; results come below.
                        if isinstance(event.item, FunctionCall):
                            yield event.item
                    case "agentia.input_item.added":
                        if isinstance(event.item, FunctionCallOutput):
                            yield event.item
                    case "response.output_text.delta":
                        yield MessageStream(_text_stream(gen, event.delta))
                    case (
                        "response.reasoning_text.delta"
                        | "response.reasoning_summary_text.delta"
                    ):
                        yield ReasoningStream(_reasoning_stream(gen, event.delta))
            await self._on_finish()

        self.__gen = __gen()

    def __aiter__(
        self,
    ) -> AsyncGenerator[
        ReasoningStream | MessageStream | FunctionCall | FunctionCallOutput, None
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


class RunEvents(RunStreamBase):
    """A streaming `Agent.run()`, yielding raw Responses API stream events."""

    def __init__(self, gen: AsyncGenerator[StreamEvent, None], agent: "Agent"):
        super().__init__(agent)

        async def __gen():
            async for event in gen:
                yield event
            await self._on_finish()

        self.__gen = __gen()

    def __aiter__(self) -> AsyncGenerator[StreamEvent, None]:
        return self.__gen

    async def __wait_for_completion(self) -> Message:
        async for _ in self.__gen:
            ...
        return last_assistant_message(self.new_items)

    def __await__(self):
        return self.__wait_for_completion().__await__()
