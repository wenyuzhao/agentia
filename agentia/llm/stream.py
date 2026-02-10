from typing import AsyncGenerator, Literal, Optional, TYPE_CHECKING

from agentia.llm.completion import Listeners, async_gen_to_sync
from agentia.spec import *

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


class ChatCompletionStreamBase:
    def __init__(self, agent: "Agent"):
        self.usage = Usage()
        self.finish_reason: FinishReason | None = None
        self.new_messages: list[NonSystemMessage] = []
        self.on_finish = Listeners()
        self.agent = agent

    def _on_finish(self):
        self.on_finish.emit()
        if self.agent:
            self.agent.emit("finish")

    def _add_new_message(self, msg: NonSystemMessage):
        self.new_messages.append(msg)


class ChatCompletionStream(ChatCompletionStreamBase):
    def __init__(self, gen: AsyncGenerator[StreamPart, None], agent: "Agent"):
        super().__init__(agent)

        async def __gen() -> (
            AsyncGenerator[
                ReasoningStream | MessageStream | ToolCall | ToolResult, None
            ]
        ):
            async for part in gen:
                if isinstance(part, (ToolCall, ToolResult)):
                    yield part
                elif isinstance(part, StreamPartTextStart):

                    async def gen2():
                        async for part2 in gen:
                            if isinstance(part2, StreamPartTextDelta):
                                yield part2.delta
                            elif isinstance(part2, StreamPartTextEnd):
                                break

                    yield MessageStream(gen2())
                elif isinstance(part, StreamPartReasoningStart):

                    async def gen2():
                        async for part2 in gen:
                            if isinstance(part2, StreamPartReasoningDelta):
                                yield part2.delta
                            elif isinstance(part2, StreamPartReasoningEnd):
                                break

                    yield ReasoningStream(gen2())
            self._on_finish()

        self.__gen = __gen()

    def __aiter__(
        self,
    ) -> AsyncGenerator[
        ReasoningStream | MessageStream | ToolCall | ToolResult | Annotation, None
    ]:
        return self.__gen

    async def __wait_for_completion(self) -> AssistantMessage:
        async for item in self.__gen:
            ...
        m = self.new_messages[-1]
        assert isinstance(m, AssistantMessage)
        return m

    def __await__(self):
        return self.__wait_for_completion().__await__()

    def __iter__(self):
        return async_gen_to_sync(self.__gen)


class ChatCompletionEvents(ChatCompletionStreamBase):
    def __init__(self, gen: AsyncGenerator[StreamPart, None], agent: "Agent"):
        super().__init__(agent)

        async def __gen():
            async for part in gen:
                yield part
            self._on_finish()

        self.__gen = __gen()

    def __aiter__(self) -> AsyncGenerator[StreamPart, None]:
        return self.__gen

    async def __wait_for_completion(self) -> AssistantMessage:
        async for item in self.__gen:
            ...
        m = self.new_messages[-1]
        assert isinstance(m, AssistantMessage)
        return m

    def __await__(self):
        return self.__wait_for_completion().__await__()
