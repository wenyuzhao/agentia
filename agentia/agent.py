import os
from pathlib import Path
from typing import Literal, Sequence, Union, overload

import uuid

from agentia.history import History
from agentia.llm import LLM, GenerationOptions
from agentia.llm.completion import ChatCompletion
from agentia.llm.stream import ChatCompletionStream, ChatCompletionEvents
from agentia.spec.chat import AssistantMessage, ToolMessage
from agentia.tools.tools import Tool, ToolSet
from agentia.spec import (
    NonSystemMessage,
    UserMessage,
    MessagePartText,
    ObjectType,
)
import logging


class Agent:
    def __init__(
        self,
        model: str | None = None,
        tools: Sequence[Tool] | None = None,
        id: str | None = None,
        name: str | None = None,
        icon: str | None = None,
        description: str | None = None,
        instructions: str | None = None,
        options: GenerationOptions | None = None,
    ) -> None:
        self.id = str(id or uuid.uuid4())
        self.name = name
        self.icon = icon
        self.description = description
        options = options or GenerationOptions()
        if tools:
            options["tools"] = ToolSet(tools)
        self.options = options
        if not model:
            model = os.getenv("AGENTIA_DEFAULT_MODEL", "openai/gpt-5-mini")
        self.llm = LLM(model)
        self.llm._agent = self
        self.history = History()
        if instructions:
            self.history.add_instructions(instructions)
        self.log = logging.getLogger(f"agentia.agent")

    def __add_prompt(
        self,
        prompt: str | NonSystemMessage | Sequence[NonSystemMessage],
    ) -> None:
        if isinstance(prompt, str):
            self.history.add(UserMessage(content=[MessagePartText(text=prompt)]))
        elif not isinstance(prompt, (list, Sequence)):
            self.history.add(prompt)
        else:
            self.history.add(*prompt)

    @overload
    def run(
        self,
        prompt: str | NonSystemMessage | Sequence[NonSystemMessage],
        /,
        stream: Literal[False] = False,
        events: Literal[False] = False,
    ) -> ChatCompletion: ...

    @overload
    def run(
        self,
        prompt: str | NonSystemMessage | Sequence[NonSystemMessage],
        /,
        stream: Literal[True],
        events: Literal[False] = False,
    ) -> ChatCompletionStream: ...

    @overload
    def run(
        self,
        prompt: str | NonSystemMessage | Sequence[NonSystemMessage],
        /,
        stream: Literal[True],
        events: Literal[True],
    ) -> ChatCompletionEvents: ...

    def run(
        self,
        prompt: str | NonSystemMessage | Sequence[NonSystemMessage],
        /,
        stream: bool = False,
        events: bool = False,
    ) -> ChatCompletion | ChatCompletionStream | ChatCompletionEvents:
        self.__add_prompt(prompt)
        if stream:
            x = self.llm.stream(self.history.get(), events=events, options=self.options)
        else:
            assert not events, "events=True is only supported with stream=True"
            x = self.llm.generate(self.history.get(), options=self.options)

        def on_finish():
            for m in x.new_messages:
                assert isinstance(m, (AssistantMessage, ToolMessage, UserMessage))
                self.history.add(m)

        x.on_finish.on(on_finish)

        return x

    async def generate_object[T: ObjectType](
        self,
        prompt: str | NonSystemMessage | Sequence[NonSystemMessage],
        type: type[T],
    ) -> T:
        self.__add_prompt(prompt)
        msgs = await self.llm._generate_object_impl(
            self.history.get(), type, options=self.options
        )

        for m in msgs:
            assert isinstance(m, (AssistantMessage, ToolMessage, UserMessage))
            self.history.add(m)

        assert isinstance(msgs[-1], AssistantMessage)

        return msgs[-1].parse(type)
