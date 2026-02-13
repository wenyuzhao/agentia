import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Optional, Sequence, overload, Any
import logging
import uuid
from agentia.history import History
from agentia.llm import LLMOptions, LLMOptionsUnion, get_provider
from agentia.llm.agentic import run_agent_loop, run_agent_loop_streamed
from agentia.llm.completion import ChatCompletion
from agentia.llm.stream import ChatCompletionStream, ChatCompletionEvents
from agentia.spec.chat import ResponseFormatJson
from agentia.tools.tools import Tool, ToolSet
from agentia.spec import (
    NonSystemMessage,
    UserMessage,
    MessagePartText,
    ObjectType,
    UserConsent,
)
from dataclasses import asdict
from agentia.tools.mcp import MCPContext
from agentia.utils.event_emitter import EventEmitter

if TYPE_CHECKING:
    from agentia.plugins.skills import Skills


class AgentEvents:
    def __init__(self):
        self.end_of_turn = EventEmitter[Callable[[], Any]]()
        self.user_consent = EventEmitter[Callable[[UserConsent], bool | None]]()


class Agent:
    def __init__(
        self,
        model: str | None = None,
        tools: Sequence[Tool] | None = None,
        id: str | None = None,
        instructions: str | Callable[[], str | None] | None = None,
        options: LLMOptionsUnion | None = None,
        skills: Sequence[Path | str] | "Skills" | bool = False,
    ) -> None:
        from agentia.plugins.skills import Skills

        self.id = str(id or uuid.uuid4())
        self.options = (
            LLMOptions(**options)
            if isinstance(options, dict)
            else (options or LLMOptions())
        )
        if skills is not False:
            tools = tools or []
            if any(isinstance(t, Skills) for t in tools):
                raise ValueError("Multiple Skills plugins provided.")
            if skills is True:
                tools = list(tools) + [Skills()]
            elif isinstance(skills, Skills):
                tools = list(tools) + [skills]
            elif isinstance(skills, Sequence):
                tools = list(tools) + [Skills(search_paths=skills)]
        self.tools = ToolSet(tools or [], self)
        if not model:
            model = os.getenv("AGENTIA_DEFAULT_MODEL", "openai/gpt-5-mini")
        self.model = model
        self.provider = get_provider(model)
        self.history = History()
        if instructions:
            self.history.add_instructions(instructions)
        self.add_instructions(self.tools.get_instructions)
        self.log = logging.getLogger(f"agentia.agent")
        self._mcp_context: Optional[MCPContext] = None
        self._temp_mcp_context: Optional[MCPContext] = None
        self.events = AgentEvents()

    def add_instructions(self, instructions: str | Callable[[], str | None]) -> None:
        self.history.add_instructions(instructions)

    def __add_prompt(
        self, prompt: str | NonSystemMessage | Sequence[NonSystemMessage]
    ) -> None:
        if isinstance(prompt, str):
            self.history.add(UserMessage(content=[MessagePartText(text=prompt)]))
        elif not isinstance(prompt, (list, Sequence)):
            self.history.add(prompt)
        else:
            self.history.add(*prompt)

    def __merge_options(self, options: LLMOptionsUnion | None) -> LLMOptions:
        options_merged = LLMOptions()
        for k, v in asdict(self.options).items():
            setattr(options_merged, k, v)
        if options:
            options_dict = (
                asdict(options) if isinstance(options, LLMOptions) else options
            )
            for k, v in options_dict.items():
                setattr(options_merged, k, v)
        return options_merged

    @overload
    def run(
        self,
        prompt: str | NonSystemMessage | Sequence[NonSystemMessage],
        *,
        stream: Literal[False] = False,
        events: Literal[False] = False,
        options: LLMOptionsUnion | None = None,
    ) -> ChatCompletion: ...

    @overload
    def run(
        self,
        prompt: str | NonSystemMessage | Sequence[NonSystemMessage],
        *,
        stream: Literal[True],
        events: Literal[False] = False,
        options: LLMOptionsUnion | None = None,
    ) -> ChatCompletionStream: ...

    @overload
    def run(
        self,
        prompt: str | NonSystemMessage | Sequence[NonSystemMessage],
        *,
        stream: Literal[True],
        events: Literal[True],
        options: LLMOptionsUnion | None = None,
    ) -> ChatCompletionEvents: ...

    def run(
        self,
        prompt: str | NonSystemMessage | Sequence[NonSystemMessage],
        *,
        stream: bool = False,
        events: bool = False,
        options: LLMOptionsUnion | None = None,
    ) -> ChatCompletion | ChatCompletionStream | ChatCompletionEvents:
        self.__add_prompt(prompt)
        options_merged = self.__merge_options(options)
        if stream:
            x = run_agent_loop_streamed(self, events, options_merged)
        else:
            assert not events, "events=True is only supported with stream=True"
            x = run_agent_loop(self, options_merged)
        return x

    async def generate_object[T: ObjectType](
        self,
        prompt: str | NonSystemMessage | Sequence[NonSystemMessage],
        type: type[T],
        options: LLMOptionsUnion | None = None,
    ) -> T:
        self.__add_prompt(prompt)
        options_merged = self.__merge_options(options)
        options_merged.response_format = ResponseFormatJson.from_model(type)
        result_msg = await run_agent_loop(self, options_merged)
        return result_msg.parse(type)

    async def __aenter__(self):
        self._mcp_context = MCPContext()
        await self._mcp_context.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        assert self._mcp_context is not None
        await self._mcp_context.__aexit__(exc_type, exc_val, exc_tb)
        self._mcp_context = None

    async def user_consent(
        self, message: str | UserConsent, details: str | None = None
    ) -> bool | None:
        if isinstance(message, str):
            consent = UserConsent(message=message, details=details)
        else:
            assert details is None
            consent = message
        if len(self.events.user_consent) > 0:
            results = await self.events.user_consent.emit(consent)
            # All handlers must return True to allow the action
            for r in results:
                if not isinstance(r, bool) or r is False:
                    return False
            return True
        # Allow any requests by default
        return True
