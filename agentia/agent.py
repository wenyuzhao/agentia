from enum import Enum
from pathlib import Path
from typing import Literal, Sequence, Union, overload

from unittest import result
import uuid

from pydantic import BaseModel
from agentia.history import History
from agentia.llm import LLM, GenerationOptions
from agentia.llm.completion import ChatCompletion
from agentia.llm.stream import ChatCompletionStream, ChatCompletionEvents
from agentia.tools.tools import Tool, ToolSet
from agentia.spec import (
    NonSystemMessage,
    SystemMessage,
    UserMessage,
    MessagePartText,
    ObjectType,
)
import logging


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
        self.history = History()
        if instructions:
            self.history.add_instructions(instructions)
        self.log = logging.getLogger(f"agentia.agent")

    @staticmethod
    def from_config(config: Union[str, Path, "Config"]) -> "Agent":
        from agentia.utils.config import load_agent_from_config

        return load_agent_from_config(config)

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
            assert not isinstance(x.messages[-1], SystemMessage)
            self.history.add(x.messages[-1])

        x.on_finish.on(on_finish)

        return x

    async def generate_object[T: ObjectType](
        self,
        prompt: str | NonSystemMessage | Sequence[NonSystemMessage],
        type: type[T],
    ) -> T:
        self.__add_prompt(prompt)
        msg = await self.llm._generate_object_impl(
            self.history.get(), type, options=self.options
        )

        self.history.add(msg)

        return msg.parse(type)


from .utils.config import Config
