import os
from typing import (
    TYPE_CHECKING,
    AsyncGenerator,
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
from agentia.spec.stream import StreamPart
from agentia.tools.tools import Tool, ToolSet
from agentia.spec import (
    NonSystemMessage,
    UserMessage,
    MessagePartText,
    ObjectType,
    ToolCall,
    UserConsentRequest,
)
from agentia.tools.mcp import MCPContext
from agentia.utils.event_emitter import EventEmitter
from pathlib import Path

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
        live_options: LiveOptions | None = None,
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
        self.live_options = live_options or LiveOptions()
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

    def __add_prompt(
        self, prompt: str | NonSystemMessage | Sequence[NonSystemMessage]
    ) -> None:
        if isinstance(prompt, str):
            self.history.add(UserMessage(content=[MessagePartText(text=prompt)]))
        elif not isinstance(prompt, (list, Sequence)):
            self.history.add(prompt)
        else:
            self.history.add(*prompt)

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
        options: LLMOptions | None = None,
    ) -> T:
        self.__add_prompt(prompt)
        options_merged = self.__merge_options(options)
        options_merged.response_format = ResponseFormatJson.from_model(type)
        result_msg = await run_agent_loop(self, options_merged)
        return result_msg.parse(type)

    async def __aenter__(self):
        self._mcp_context = MCPContext()
        await self._mcp_context.__aenter__()
        # Connect live session if provider supports it
        if self.provider.supports_live:
            await self.tools.init()
            instructions = self.history.get_instructions() or None
            await self.provider.connect_live(
                self.live_options, self.tools, instructions
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Disconnect live session if provider supports it
        if self.provider.supports_live:
            await self.provider.disconnect_live()
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

    # --- Live session methods ---

    async def send_audio(
        self, data: bytes, mime_type: str = "audio/pcm;rate=16000"
    ) -> None:
        """Send an audio chunk to the live session. Default: PCM 16kHz 16-bit mono."""
        await self.provider.send_audio(data, mime_type)

    async def send_video(
        self, data: bytes, mime_type: str = "image/jpeg"
    ) -> None:
        """Send a video frame to the live session."""
        await self.provider.send_video(data, mime_type)

    async def send_text(self, text: str) -> None:
        """Send text input to the live session."""
        await self.provider.send_text_live(text)

    async def send_audio_stream_end(self) -> None:
        """Signal end of audio stream to flush cached audio."""
        await self.provider.send_audio_stream_end()

    async def receive(self) -> AsyncGenerator[StreamPart, None]:
        """Receive stream parts from the live session.

        When auto_tool_execution is enabled (default), tool calls are
        automatically executed and responses sent back to the model.
        """
        async for event in self.provider.receive():
            if (
                isinstance(event, ToolCall)
                and self.live_options.auto_tool_execution
            ):
                yield event
                # Auto-execute the tool
                responses = await self.tools.run(self, [event], parallel=False)
                for resp in responses:
                    # Send response back to the model
                    await self.provider.send_tool_response(
                        resp.tool_call_id, resp.output
                    )
                    yield resp
            else:
                yield event
