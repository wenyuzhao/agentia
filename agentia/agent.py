import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence, overload
import logging
import uuid
from agentia.history import History
from agentia.llm import LLM, LLMOptions, LLMOptionsUnion
from agentia.llm.completion import ChatCompletion
from agentia.llm.stream import ChatCompletionStream, ChatCompletionEvents
from agentia.plugins import skills
from agentia.spec.chat import AssistantMessage, ToolMessage
from agentia.tools.tools import Tool, ToolSet
from agentia.spec import NonSystemMessage, UserMessage, MessagePartText, ObjectType
from dataclasses import asdict
from agentia.tools.mcp import MCPContext
from agentia.tools.plugin import Plugin

if TYPE_CHECKING:
    from agentia.plugins.skills import Skills


class Agent:
    def __init__(
        self,
        model: str | None = None,
        tools: Sequence[Tool] | None = None,
        id: str | None = None,
        instructions: str | None = None,
        options: LLMOptionsUnion | None = None,
        skills: Sequence[Path | str] | "Skills" | bool | None = None,
    ) -> None:
        from agentia.plugins.skills import Skills

        self.id = str(id or uuid.uuid4())
        self.options = (
            LLMOptions(**options)
            if isinstance(options, dict)
            else (options or LLMOptions())
        )
        if skills:
            if not tools:
                tools = []
            if any(isinstance(t, Skills) for t in tools):
                raise ValueError(
                    "Cannot add a Skills plugin when another Skills plugin is already added"
                )
            if skills is True:
                tools = list(tools) + [Skills()]
            elif isinstance(skills, Skills):
                tools = list(tools) + [skills]
            elif isinstance(skills, Sequence):
                tools = list(tools) + [Skills(search_paths=skills)]
        if tools:
            self.options.tools = ToolSet(tools)
        if not model:
            model = os.getenv("AGENTIA_DEFAULT_MODEL", "openai/gpt-5-mini")
        self.llm = LLM(model)
        self.llm._agent = self
        self.history = History()
        if instructions:
            self.history.add_instructions(instructions)
        if self.options.tools and isinstance(self.options.tools, ToolSet):
            self.history.add_instructions(self.options.tools.get_instructions())
        self.log = logging.getLogger(f"agentia.agent")
        self.__mcp_context = MCPContext()

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
        options: LLMOptionsUnion | None = None,
    ) -> ChatCompletion: ...

    @overload
    def run(
        self,
        prompt: str | NonSystemMessage | Sequence[NonSystemMessage],
        /,
        stream: Literal[True],
        events: Literal[False] = False,
        options: LLMOptionsUnion | None = None,
    ) -> ChatCompletionStream: ...

    @overload
    def run(
        self,
        prompt: str | NonSystemMessage | Sequence[NonSystemMessage],
        /,
        stream: Literal[True],
        events: Literal[True],
        options: LLMOptionsUnion | None = None,
    ) -> ChatCompletionEvents: ...

    def run(
        self,
        prompt: str | NonSystemMessage | Sequence[NonSystemMessage],
        /,
        stream: bool = False,
        events: bool = False,
        options: LLMOptionsUnion | None = None,
    ) -> ChatCompletion | ChatCompletionStream | ChatCompletionEvents:
        self.__add_prompt(prompt)
        options_merged = LLMOptions()
        for k, v in asdict(self.options).items():
            setattr(options_merged, k, v)
        if options:
            options_dict = (
                asdict(options) if isinstance(options, LLMOptions) else options
            )
            for k, v in options_dict.items():
                setattr(options_merged, k, v)
        if stream:
            x = self.llm.stream(
                self.history.get(), events=events, options=options_merged
            )
        else:
            assert not events, "events=True is only supported with stream=True"
            x = self.llm.generate(self.history.get(), options=options_merged)

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

    async def __aenter__(self):
        await self.__mcp_context.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.__mcp_context.__aexit__(exc_type, exc_val, exc_tb)
