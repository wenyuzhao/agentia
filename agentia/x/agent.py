from typing import Literal, Sequence, overload
from agentia.llm import LLM
from agentia.llm.completion import ChatCompletion
from agentia.llm.stream import ChatCompletionStream
from agentia.spec import Message, UserMessage, MessagePartText


class Agent:
    def __init__(self, model: str) -> None:
        self.llm = LLM(model)
        self.history: list[Message] = []

    async def init(self): ...

    @overload
    def run(
        self,
        prompt: str | Message | Sequence[Message],
        /,
        stream: Literal[False] = False,
    ) -> ChatCompletion: ...

    @overload
    def run(
        self, prompt: str | Message | Sequence[Message], /, stream: Literal[True]
    ) -> ChatCompletionStream: ...

    def run(
        self, prompt: str | Message | Sequence[Message], /, stream: bool = False
    ) -> ChatCompletion | ChatCompletionStream:
        if isinstance(prompt, str):
            self.history.append(UserMessage(content=[MessagePartText(text=prompt)]))
        elif not isinstance(prompt, (list, Sequence)):
            self.history.append(prompt)
        else:
            self.history.extend(prompt)

        if stream:
            x = self.llm.stream(self.history)
        else:
            x = self.llm.generate(self.history)

        def on_finish():
            self.history.append(x.messages[-1])

        x.on_finish.on(on_finish)

        return x
