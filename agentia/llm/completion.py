from typing import AsyncGenerator

from agentia.spec.base import FinishReason, Usage, Warning
from agentia.spec.prompt import AssistantMessage, Message


class Listeners:
    def __init__(self):
        self.__listeners = []

    def on(self, listener):
        self.__listeners.append(listener)

    def off(self, listener):
        self.__listeners.remove(listener)

    def emit(self):
        for listener in self.__listeners:
            listener()


class ChatCompletion:
    def __init__(
        self,
        gen: AsyncGenerator[AssistantMessage, None],
    ):
        async def __gen() -> AsyncGenerator[AssistantMessage, None]:
            async for msg in gen:
                yield msg
            self.on_finish.emit()

        self.__gen = __gen()
        self.usage = Usage()
        self.finish_reason: FinishReason | None = None
        self.warnings: list[Warning] = []
        self.messages: list[Message] = []
        self.on_finish = Listeners()

    def __aiter__(self) -> AsyncGenerator[AssistantMessage, None]:
        return self.__gen

    async def __wait_for_completion(self) -> AssistantMessage:
        async for msg in self.__gen:
            ...
        m = self.messages[-1]
        assert isinstance(m, AssistantMessage)
        return m

    def __await__(self):
        return self.__wait_for_completion().__await__()
