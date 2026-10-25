import inspect
import logging
import os
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import httpx

from agentia.compaction import CompactionOptions, compaction_input, render_items
from agentia.history import History, normalize_tool_pairs
from agentia.llm import LLMOptions, get_provider
from agentia.llm.completion import Run, RunBase
from agentia.llm.loop import StepInfo, new_call_id, run_loop
from agentia.llm.providers import ModelRequest, Provider
from agentia.llm.stream import RunEvents, RunStream
from agentia.models import (
    Compaction,
    FunctionCall,
    FunctionCallOutput,
    InputItem,
    Instruction,
    Item,
    ObjectType,
    ResponseInfo,
    Usage,
    UserConsentRequest,
)
from agentia.models.options import ResponseFormatJson
from agentia.models.stream import StreamEvent
from agentia.tools.tools import BaseTool, Suspend, Tool, ToolResult, ToolSet
from agentia.utils.commands import Commands
from agentia.utils.event_emitter import EventEmitter

if TYPE_CHECKING:
    from agentia.plugins.skills import Skills

type InstructionSource = (
    str
    | Instruction
    | Callable[[], str | Instruction | None | Awaitable[str | Instruction | None]]
)

DEFAULT_CONTEXT_WINDOW = 128_000


class AgentEvents:
    def __init__(self):
        self.end_of_turn = EventEmitter[Callable[[], Any]]()


class Agent:
    """An LLM agent: a model, tools, instructions and a history.

    Subclass and override the hooks (`poll_inputs`, `before_model_call`,
    `prepare_input`, `on_response`, `before_tool_call`, `after_tool_call`,
    `on_compaction`, ...) to integrate with a host application.
    """

    def __init__(
        self,
        model: str | None = None,
        tools: Sequence[Tool] | None = None,
        id: str | None = None,
        instructions: InstructionSource | Sequence[InstructionSource] | None = None,
        options: LLMOptions | None = None,
        skills: "Sequence[Path | str] | Skills | bool" = False,
        *,
        provider: Provider | None = None,
        history: History | None = None,
        compaction: CompactionOptions | bool | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """
        - `model`: a `[provider:][owner/]model[:think]` selector (ignored with `provider`).
        - `provider`: a configured provider (endpoint, key, headers, HTTP client).
        - `history`: the conversation store; subclass `History` to persist it.
        - `compaction`: `True` / `CompactionOptions` to compact automatically.
        - `client`: a shared `httpx.AsyncClient` for model calls.
        """
        from agentia.plugins.skills import Skills

        self.__id = str(id or uuid.uuid4())
        self.options = (
            LLMOptions(**options)
            if isinstance(options, dict)
            else (options or LLMOptions())
        )
        if skills is not False:
            tools = list(tools or [])
            if any(isinstance(t, Skills) for t in tools):
                raise ValueError("Multiple Skills plugins provided.")
            if skills is True:
                tools.append(Skills())
            elif isinstance(skills, Skills):
                tools.append(skills)
            elif isinstance(skills, Sequence):
                tools.append(Skills(search_paths=skills))
        self.http_client = client
        if provider is None:
            if not model:
                model = os.getenv("AGENTIA_DEFAULT_MODEL", "openai/gpt-5-mini")
            provider = get_provider(model, client=client)
        self.provider = provider
        self.history = history if history is not None else History()
        self.compaction: CompactionOptions | None = (
            CompactionOptions() if compaction is True else (compaction or None)
        )
        self.tools = ToolSet(tools or [], self)
        self._instructions: list[InstructionSource] = []
        if instructions:
            if isinstance(instructions, (str, Instruction)) or callable(instructions):
                self._instructions.append(instructions)
            else:
                self._instructions.extend(instructions)
        self.log = logging.getLogger("agentia.agent")
        self._in_context = False
        self.events = AgentEvents()
        self.commands = Commands(self)
        self.__deferred_items: list[Item] = []
        self.env: dict[str, str] = {}
        """Session-local environment variables that can be used in tools. This is also passed to the processes launched by the Bash tool."""

    @property
    def id(self) -> str:
        return self.__id

    @property
    def model(self) -> str:
        return self.provider.model

    @property
    def current_context_length(self) -> int:
        return self.history.current_tokens

    @property
    def usage(self) -> Usage:
        return self.history.usage

    async def get_max_context_length(self) -> int:
        """Return the context length of the model, delegating to the provider."""
        return await self.provider.get_context_length()

    # ----- instructions -----

    def add_instructions(self, instructions: InstructionSource) -> None:
        self._instructions.append(instructions)

    async def get_instructions(self) -> list[Instruction]:
        """The system prompt blocks: the agent's own, then plugins' and tool providers'."""
        out: list[Instruction] = []
        for src in self._instructions:
            v = src() if callable(src) else src
            if inspect.isawaitable(v):
                v = await v
            if isinstance(v, Instruction):
                if v.text.strip():
                    out.append(v)
            elif v and v.strip():
                out.append(Instruction(text=v))
        out += [Instruction(text=t) for t in await self.tools.get_instructions()]
        return out

    def render_summary(self, compaction: Compaction) -> str:
        """How the latest compaction summary appears in the system prompt."""
        return f"# Summary of the conversation so far\n\n{compaction.summary}"

    async def build_request(self, options: LLMOptions | None = None) -> ModelRequest:
        c = self.history.compaction
        return ModelRequest(
            items=await self.prepare_input(self.history.context()),
            instructions=await self.get_instructions(),
            summary=self.render_summary(c) if c else None,
            tools=self.tools.get_schema(),
            options=options or self.options,
        )

    # ----- hooks -----

    async def poll_inputs(self, after_answer: bool = False) -> Sequence[Item]:
        """New input to append before each model call. Also called after a final
        answer (`after_answer=True`): returning items then continues the run, so input
        that arrived while the model answered is not left waiting. Default: items
        queued with `defer()`."""
        return self.clear_deferred_items()

    async def before_model_call(self, step: StepInfo) -> None:
        """Called before each model call; raise `StopRun` to end the run (budgets,
        access checks, ...)."""

    async def prepare_input(self, items: list[Item]) -> list[Item]:
        """Transform the context items before they are sent (e.g. shorten old tool
        results, attach images). The default pairs every call with its output."""
        return normalize_tool_pairs(items)

    async def on_response(self, items: list[Item], response: ResponseInfo) -> None:
        """Called after a response's items were appended to the history."""

    async def before_tool_call(
        self, call: FunctionCall, tool: BaseTool | None
    ) -> ToolResult | FunctionCallOutput | Suspend | None:
        """Called before a pending call runs (also when a run resumes). Return `None`
        to run it, a result to use instead, or `Suspend()` to pause the run (e.g. to
        ask the user for approval)."""
        return None

    async def after_tool_call(
        self, call: FunctionCall, output: FunctionCallOutput
    ) -> FunctionCallOutput:
        """Called with each output before it is appended; may return a replacement
        (e.g. a shortened output with the full text saved elsewhere)."""
        return output

    async def on_compaction(self, compaction: Compaction, replaced: list[Item]) -> None:
        """Called before a compaction is committed (raise to abort it)."""

    async def render_compaction_input(
        self, previous: Compaction | None, items: list[Item], focus: str | None
    ) -> str:
        """The summarizer's input: the previous summary and the items to fold in."""
        return compaction_input(previous, render_items(items), focus)

    def is_turn_start(self, item: Item) -> bool:
        """Where a user↔assistant exchange starts, for compaction's split. Default: any
        user message. Hosts that inject user-role notices can exclude them here."""
        from agentia.compaction import is_user_turn

        return is_user_turn(item)

    async def context_window(self) -> int:
        """The model's context window, for the compaction threshold."""
        try:
            return await self.provider.get_context_length()
        except Exception as e:  # noqa: BLE001 -- fall back to a default window
            self.log.warning("context length of %s unknown: %s", self.model, e)
            return DEFAULT_CONTEXT_WINDOW

    def new_call_id(self, call: FunctionCall, seen: set[str]) -> str:
        """The call id to record for a new function call; `seen` holds the ids already
        in the history. Default: the provider's id unless it is empty or taken."""
        return new_call_id(call, seen)

    # ----- running -----

    def __merge_options(self, options: LLMOptions | None) -> LLMOptions:
        if not options:
            return self.options.model_copy()
        # Only what the caller set explicitly overrides the agent's options.
        return self.options.model_copy(update=options.model_dump(exclude_unset=True))

    def _events(
        self, prompt: InputItem | None, options: LLMOptions
    ) -> Callable[[RunBase], AsyncGenerator[StreamEvent, None]]:
        def make(run: RunBase) -> AsyncGenerator[StreamEvent, None]:
            async def gen() -> AsyncGenerator[StreamEvent, None]:
                try:
                    async for e in run_loop(self, prompt, options, run):
                        yield e
                finally:
                    if not self._in_context:
                        await self.tools.aclose()

            return gen()

        return make

    @overload
    def run(
        self,
        prompt: InputItem | None,
        *,
        stream: Literal[False] = False,
        events: Literal[False] = False,
        options: LLMOptions | None = None,
    ) -> Run: ...

    @overload
    def run(
        self,
        prompt: InputItem | None,
        *,
        stream: Literal[True],
        events: Literal[False] = False,
        options: LLMOptions | None = None,
    ) -> RunStream: ...

    @overload
    def run(
        self,
        prompt: InputItem | None,
        *,
        stream: Literal[True],
        events: Literal[True],
        options: LLMOptions | None = None,
    ) -> RunEvents: ...

    def run(
        self,
        prompt: InputItem | None,
        *,
        stream: bool = False,
        events: bool = False,
        options: LLMOptions | None = None,
    ) -> Run | RunStream | RunEvents:
        """Run the agent. `prompt=None` continues from the history: it executes
        pending tool calls (e.g. after a suspension or a crash) or calls the model."""
        make = self._events(prompt, self.__merge_options(options))
        if not stream:
            assert not events, "events=True is only supported with stream=True"
            return Run(make, self)
        return RunEvents(make, self) if events else RunStream(make, self)

    async def generate_object[T: ObjectType](
        self,
        prompt: InputItem,
        type: type[T],
        options: LLMOptions | None = None,
    ) -> T:
        opts = self.__merge_options(options)
        opts.response_format = ResponseFormatJson.from_model(type)
        result_msg = await Run(self._events(prompt, opts), self)
        return result_msg.parse(type)

    async def compact(
        self,
        focus: str | None = None,
        *,
        keep_exchanges: int | None = None,
        model: str | None = None,
        effort: Literal["low", "medium", "high"] | None = None,
    ) -> Compaction | None:
        """Compact now. `keep_exchanges=0` folds the whole context into the summary.
        Returns None when there is nothing old enough to compact."""
        from agentia.compaction import compact

        opts = (self.compaction or CompactionOptions()).model_copy()
        if model:
            opts.model = model
        if effort:
            focus = "\n".join(
                x
                for x in [
                    focus,
                    {
                        "low": "Keep as much important information as possible.",
                        "medium": "Drop routine details, verbose outputs and redundant exchanges.",
                        "high": "Be very aggressive: keep only the most critical information.",
                    }[effort],
                ]
                if x
            )
        return await compact(
            self, focus=focus, keep_exchanges=keep_exchanges, options=opts
        )

    # ----- lifecycle -----

    async def __aenter__(self):
        self._in_context = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._in_context = False
        await self.aclose()

    async def aclose(self) -> None:
        """Close MCP connections and other tool resources."""
        await self.tools.aclose()

    # ----- human in the loop -----

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

    async def defer(self, item: Item) -> None:
        """
        Enqueue an item to be added to the history before the next model call (e.g.
        from a tool, in the middle of a run).
        """
        self.__deferred_items.append(item)

    def has_deferred_items(self) -> bool:
        return len(self.__deferred_items) > 0

    def clear_deferred_items(self) -> list[Item]:
        items = self.__deferred_items
        self.__deferred_items = []
        return items
