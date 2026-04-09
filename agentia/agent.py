import os
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    Optional,
    Sequence,
    overload,
    Any,
)
import logging
import uuid
from agentia.history import History
from agentia.live import LiveOptions
from agentia.llm import LLMOptions, get_provider
from agentia.llm.agentic import run_agent_loop, run_agent_loop_streamed
from agentia.llm.completion import ChatCompletion
from agentia.llm.stream import ChatCompletionStream, ChatCompletionEvents
from agentia.spec.chat import ResponseFormatJson
from agentia.tools.tools import Tool, ToolSet
from agentia.spec import (
    NonSystemMessage,
    ObjectType,
    UserConsentRequest,
)
from agentia.tools.mcp import MCPContext
from agentia.utils.event_emitter import EventEmitter
from pathlib import Path
from agentia.live import Live

if TYPE_CHECKING:
    from agentia.plugins.skills import Skills


class AgentEvents:
    def __init__(self):
        self.end_of_turn = EventEmitter[Callable[[], Any]]()


class Agent:
    def __init__(
        self,
        model: str | None = None,
        tools: Sequence[Tool] | None = None,
        id: str | None = None,
        instructions: (
            str
            | Callable[[], str | None]
            | Sequence[str | Callable[[], str | None]]
            | None
        ) = None,
        options: LLMOptions | None = None,
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
            if isinstance(instructions, (str, Callable)):
                self.history.add_instructions(instructions)
            else:
                for i in instructions:
                    self.history.add_instructions(i)
        self.add_instructions(self.tools.get_instructions)
        self.log = logging.getLogger(f"agentia.agent")
        self._mcp_context: Optional[MCPContext] = None
        self._temp_mcp_context: Optional[MCPContext] = None
        self.events = AgentEvents()

    def add_instructions(self, instructions: str | Callable[[], str | None]) -> None:
        self.history.add_instructions(instructions)

    def __merge_options(self, options: LLMOptions | None) -> LLMOptions:
        options_merged = {}
        for k, v in self.options.model_dump().items():
            options_merged[k] = v
        if options:
            for k, v in options.model_dump().items():
                options_merged[k] = v
        return LLMOptions(**options_merged)

    @overload
    def run(
        self,
        prompt: str | NonSystemMessage | Sequence[NonSystemMessage],
        *,
        stream: Literal[False] = False,
        events: Literal[False] = False,
        options: LLMOptions | None = None,
    ) -> ChatCompletion: ...

    @overload
    def run(
        self,
        prompt: str | NonSystemMessage | Sequence[NonSystemMessage],
        *,
        stream: Literal[True],
        events: Literal[False] = False,
        options: LLMOptions | None = None,
    ) -> ChatCompletionStream: ...

    @overload
    def run(
        self,
        prompt: str | NonSystemMessage | Sequence[NonSystemMessage],
        *,
        stream: Literal[True],
        events: Literal[True],
        options: LLMOptions | None = None,
    ) -> ChatCompletionEvents: ...

    def run(
        self,
        prompt: str | NonSystemMessage | Sequence[NonSystemMessage],
        *,
        stream: bool = False,
        events: bool = False,
        options: LLMOptions | None = None,
    ) -> ChatCompletion | ChatCompletionStream | ChatCompletionEvents:
        options_merged = self.__merge_options(options)
        if stream:
            x = run_agent_loop_streamed(self, prompt, events, options_merged, None)
        else:
            assert not events, "events=True is only supported with stream=True"
            x = run_agent_loop(self, prompt, options_merged, None)
        return x

    async def generate_object[T: ObjectType](
        self,
        prompt: str | NonSystemMessage | Sequence[NonSystemMessage],
        type: type[T],
        options: LLMOptions | None = None,
    ) -> T:
        options_merged = self.__merge_options(options)
        options_merged.response_format = ResponseFormatJson.from_model(type)
        result_msg = await run_agent_loop(self, prompt, options_merged, None)
        return result_msg.parse(type)

    async def __aenter__(self):
        self._mcp_context = MCPContext()
        await self._mcp_context.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        assert self._mcp_context is not None
        await self._mcp_context.__aexit__(exc_type, exc_val, exc_tb)
        self._mcp_context = None

    async def user_consent(self, request: UserConsentRequest) -> bool | str | None:
        return True

    async def user_consent_guard(self, request: str | UserConsentRequest):
        if isinstance(request, str):
            request = UserConsentRequest(message=request)
        consent = await self.user_consent(request)
        if consent is True:
            return
        elif consent is False:
            raise PermissionError("User denied consent.")
        elif isinstance(consent, str):
            raise PermissionError(f"User denied consent: {consent}")
        elif consent is None:
            raise PermissionError("User did not respond to consent request.")
        else:
            raise ValueError(f"Invalid consent response: {consent}")

    def live(self, options: LiveOptions | None = None) -> Live:
        if not self.provider.supports_live:
            raise NotImplementedError("This provider does not support live sessions")
        return Live(self, options)
