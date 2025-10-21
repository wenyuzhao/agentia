from typing import Literal, Sequence, overload

import uuid
from agentia.llm import LLM, GenerationOptions
from agentia.llm.completion import ChatCompletion
from agentia.llm.stream import ChatCompletionStream, ChatCompletionEvents
from agentia.llm.tools import Tool, ToolSet
from agentia.spec import Message, UserMessage, MessagePartText
import logging

from agentia.spec.prompt import SystemMessage


class Agent:
    def __init__(
        self,
        model: str,
        tools: Sequence[Tool] | None = None,
        id: str | None = None,
        instructions: str | None = None,
        options: GenerationOptions | None = None,
    ) -> None:
        self.id = str(id or uuid.uuid4())
        options = options or GenerationOptions()
        if tools:
            options["tools"] = ToolSet(tools)
        self.options = options
        self.llm = LLM(model)
        self.llm._agent = self
        self.history: list[Message] = []
        if instructions:
            self.history.append(SystemMessage(content=instructions))
        self.log = logging.getLogger(f"agentia.agent")

    @overload
    def run(
        self,
        prompt: str | Message | Sequence[Message],
        /,
        stream: Literal[False] = False,
        events: Literal[False] = False,
    ) -> ChatCompletion: ...

    @overload
    def run(
        self,
        prompt: str | Message | Sequence[Message],
        /,
        stream: Literal[True],
        events: Literal[False] = False,
    ) -> ChatCompletionStream: ...

    @overload
    def run(
        self,
        prompt: str | Message | Sequence[Message],
        /,
        stream: Literal[True],
        events: Literal[True],
    ) -> ChatCompletionEvents: ...

    def run(
        self,
        prompt: str | Message | Sequence[Message],
        /,
        stream: bool = False,
        events: bool = False,
    ) -> ChatCompletion | ChatCompletionStream | ChatCompletionEvents:
        if isinstance(prompt, str):
            self.history.append(UserMessage(content=[MessagePartText(text=prompt)]))
        elif not isinstance(prompt, (list, Sequence)):
            self.history.append(prompt)
        else:
            self.history.extend(prompt)

        if stream:
            x = self.llm.stream(self.history, events=events, options=self.options)
        else:
            assert not events, "events=True is only supported with stream=True"
            x = self.llm.generate(self.history, options=self.options)

        def on_finish():
            self.history.append(x.messages[-1])

        x.on_finish.on(on_finish)

        return x
