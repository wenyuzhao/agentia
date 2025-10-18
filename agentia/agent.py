from typing import Literal, Sequence, overload
from agentia.llm import LLM, GenerationOptions
from agentia.llm.completion import ChatCompletion
from agentia.llm.stream import ChatCompletionStream
from agentia.llm.tools import Tool, ToolSet
from agentia.spec import Message, UserMessage, MessagePartText
import logging


class Agent:
    def __init__(self, model: str, tools: Sequence[Tool]) -> None:
        self.llm = LLM(model)
        self.llm._agent = self
        self.history: list[Message] = []
        self.log = logging.getLogger(f"agentia.agent")
        self.__tools = ToolSet(tools)

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

        options = GenerationOptions(tools=self.__tools)

        if stream:
            x = self.llm.stream(self.history, options=options)
        else:
            x = self.llm.generate(self.history, options=options)

        def on_finish():
            self.history.append(x.messages[-1])

        x.on_finish.on(on_finish)

        return x
