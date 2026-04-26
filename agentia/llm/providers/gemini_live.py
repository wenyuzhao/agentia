import base64
import os
from typing import TYPE_CHECKING, Any, AsyncGenerator, cast, override, Optional
from uuid import uuid4

import httpx
from google import genai
from google.genai import types
from pydantic import HttpUrl

if TYPE_CHECKING:
    from agentia.history import History
    from google.genai.live import AsyncSession

from agentia.live import LiveOptions
from agentia.llm import LLMOptions
from agentia.llm.providers import GenerationResult, Provider
from agentia.models.base import FunctionTool, ToolCallResponse, Usage
from agentia.models.chat import (
    AssistantMessage,
    Message,
    MessagePartFile,
    MessagePartReasoning,
    MessagePartText,
    MessagePartToolCall,
    MessagePartToolResult,
    ToolMessage,
    UserMessage,
)
from agentia.models.base import ToolCall
from agentia.models.stream import (
    StreamPart,
    StreamPartAudioDelta,
    StreamPartAudioEnd,
    StreamPartAudioStart,
    StreamPartTextDelta,
    StreamPartTextEnd,
    StreamPartTextStart,
    StreamPartInputTranscriptionDelta,
    StreamPartInputTranscriptionEnd,
    StreamPartInputTranscriptionStart,
    StreamPartOutputTranscriptionDelta,
    StreamPartOutputTranscriptionEnd,
    StreamPartOutputTranscriptionStart,
    StreamPartTurnEnd,
    StreamPartTurnStart,
)
from agentia.tools.tools import ToolSet
from agentia.models.live import (
    LiveChunk,
    LiveChunkAudio,
    LiveChunkText,
    LiveChunkImage,
    LiveChunkVideo,
    LiveChunkEnd,
)


def _get_usage(u: types.UsageMetadata | None) -> Usage:
    if not u:
        return Usage()
    return Usage(
        input_tokens=u.prompt_token_count,
        output_tokens=u.response_token_count,
        total_tokens=u.total_token_count,
        reasoning_tokens=u.thoughts_token_count,
        cached_input_tokens=u.cached_content_token_count,
    )


def _convert_tools_to_genai(tools: ToolSet) -> list[types.Tool] | None:
    schemas = tools.get_schema()
    if not schemas:
        return None
    declarations: list[types.FunctionDeclaration] = []
    for schema in schemas:
        if isinstance(schema, FunctionTool):
            declarations.append(
                types.FunctionDeclaration(
                    name=schema.name,
                    description=schema.description,
                    parameters_json_schema=schema.input_schema,
                )
            )
    if not declarations:
        return None
    return [types.Tool(function_declarations=declarations)]


def _file_part_to_blob(part: MessagePartFile) -> types.Blob:
    """Convert a MessagePartFile's data to a Gemini Blob."""
    data = part.data
    if isinstance(data, bytes):
        return types.Blob(data=data, mime_type=part.media_type)
    elif isinstance(data, HttpUrl) or (
        isinstance(data, str) and data.startswith(("http://", "https://"))
    ):
        raise ValueError(
            "URL-based file parts are not supported for Gemini Live; provide raw bytes or base64 data."
        )
    elif isinstance(data, str):
        if data.startswith("data:"):
            # data URL: extract base64 payload
            _, payload = data.split(",", 1)
            return types.Blob(data=base64.b64decode(payload), mime_type=part.media_type)
        else:
            # Treat as base64 string
            return types.Blob(data=base64.b64decode(data), mime_type=part.media_type)
    raise ValueError(f"Unsupported file data type: {type(data)}")


def _convert_message_to_content(msg: Message) -> types.Content | None:
    """Convert an agentia Message to a google.genai types.Content for history seeding."""
    if isinstance(msg, UserMessage):
        parts: list[types.Part] = []
        if isinstance(msg.content, str):
            parts.append(types.Part(text=msg.content))
        else:
            for p in msg.content:
                if isinstance(p, MessagePartText):
                    parts.append(types.Part(text=p.text))
                elif isinstance(p, MessagePartFile):
                    blob = _file_part_to_blob(p)
                    parts.append(types.Part(inline_data=blob))
        return types.Content(role="user", parts=parts) if parts else None

    if isinstance(msg, AssistantMessage):
        parts = []
        if isinstance(msg.content, str):
            parts.append(types.Part(text=msg.content))
        else:
            for p in msg.content:
                if isinstance(p, MessagePartText):
                    parts.append(types.Part(text=p.text))
                elif isinstance(p, MessagePartReasoning):
                    parts.append(types.Part(thought=True, text=p.text))
                elif isinstance(p, MessagePartToolCall):
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                name=p.tool_name, args=p.input
                            )
                        )
                    )
        return types.Content(role="model", parts=parts) if parts else None

    if isinstance(msg, ToolMessage):
        parts = []
        for p in msg.content:
            if isinstance(p, MessagePartToolResult):
                output = p.output
                if not isinstance(output, dict):
                    output = {"result": output}
                parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=p.tool_name, response=output
                        )
                    )
                )
        return types.Content(role="user", parts=parts) if parts else None

    return None


def _convert_messages_to_contents(
    messages: list[Message],
) -> list[types.Content]:
    """Convert a list of agentia Messages to Gemini Content objects."""
    contents: list[types.Content] = []
    for msg in messages:
        content = _convert_message_to_content(msg)
        if content:
            contents.append(content)
    return contents


async def _send_user_message_realtime(
    session: "AsyncSession", msg: UserMessage
) -> None:
    """Send a UserMessage via send_realtime_input, supporting all part types."""
    if isinstance(msg.content, str):
        await session.send_realtime_input(text=msg.content)
        return

    for p in msg.content:
        if isinstance(p, MessagePartText):
            await session.send_realtime_input(text=p.text)
        elif isinstance(p, MessagePartFile):
            blob = _file_part_to_blob(p)
            if p.media_type.startswith("audio/"):
                await session.send_realtime_input(audio=blob)
            elif p.media_type.startswith(("image/", "video/")):
                await session.send_realtime_input(video=blob)
            else:
                # Fallback: send as video for other binary types
                await session.send_realtime_input(video=blob)


def _build_config(
    options: LiveOptions,
    tools: ToolSet,
    instructions: str | None,
) -> types.LiveConnectConfig:
    modalities = [
        types.Modality.AUDIO if m == "audio" else types.Modality.TEXT
        for m in options.modalities
    ]

    speech_config: types.SpeechConfig | None = None
    if options.voice or options.language:
        speech_config = types.SpeechConfig(
            voice_config=(
                types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=options.voice
                    )
                )
                if options.voice
                else None
            ),
            language_code=options.language,
        )

    thinking_config: types.ThinkingConfig | None = None
    if options.thinking_level:
        level_map = {
            "minimal": types.ThinkingLevel.MINIMAL,
            "low": types.ThinkingLevel.LOW,
            "medium": types.ThinkingLevel.MEDIUM,
            "high": types.ThinkingLevel.HIGH,
        }
        thinking_config = types.ThinkingConfig(
            thinking_level=level_map[options.thinking_level]
        )

    input_audio_transcription = types.AudioTranscriptionConfig()
    output_audio_transcription = types.AudioTranscriptionConfig()

    context_window_compression = types.ContextWindowCompressionConfig(
        sliding_window=types.SlidingWindow()
    )

    session_resumption = None
    # session_resumption = types.SessionResumptionConfig(transparent=True)

    realtime_input_config: types.RealtimeInputConfig | None = None
    if not options.vad:
        realtime_input_config = types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
        )

    genai_tools: Any = _convert_tools_to_genai(tools)

    history_config = types.HistoryConfig(initial_history_in_client_content=True)

    return types.LiveConnectConfig(
        response_modalities=modalities,
        speech_config=speech_config,
        thinking_config=thinking_config,
        system_instruction=instructions or None,
        tools=genai_tools,
        input_audio_transcription=input_audio_transcription,
        output_audio_transcription=output_audio_transcription,
        context_window_compression=context_window_compression,
        session_resumption=session_resumption,
        realtime_input_config=realtime_input_config,
        history_config=history_config,
    )


_context_length_cache: dict[str, int] = {}


class GeminiLive(Provider):
    name = "gemini-live"

    def __init__(self, model: str):
        super().__init__(name="gemini-live", model=model)
        if model != "gemini-3.1-flash-live-preview":
            raise ValueError(
                "Unsupported Gemini Live model. Currently only 'gemini-3.1-flash-live-preview' is supported."
            )
        self._session: Optional["AsyncSession"] = None
        self._session_cm: Any = None  # The async context manager from connect()
        self._client: genai.Client | None = None
        self._history: Optional["History"] = None

    def _get_client(self) -> genai.Client:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY or GEMINI_API_KEY environment variable must be set"
            )
        return genai.Client(api_key=api_key)

    @override
    async def _fetch_context_length(self) -> int:
        if self.model in _context_length_cache:
            return _context_length_cache[self.model]
        client = self._client or self._get_client()
        model_info = await client.aio.models.get(model=f"models/{self.model}")
        if model_info.input_token_limit is None:
            raise ValueError(
                f"Context length not available for model '{self.model}' from Gemini API"
            )
        _context_length_cache[self.model] = model_info.input_token_limit
        return _context_length_cache[self.model]

    @property
    @override
    def supports_live(self) -> bool:
        return True

    @override
    async def connect_live(
        self,
        options: LiveOptions,
        tools: ToolSet,
        instructions: str | None,
        history: "History",
    ) -> None:
        self._client = self._get_client()
        self._history = history
        config = _build_config(options, tools, instructions)
        self._session_cm = self._client.aio.live.connect(
            model=self.model, config=config
        )
        self._session = await self._session_cm.__aenter__()

        # Seed initial history via send_client_content
        if history:
            if initial_messages := history.get():
                contents = _convert_messages_to_contents(initial_messages)
                if contents:
                    contents = cast(list[types.Content | types.ContentDict], contents)
                    assert self._session is not None
                    await self._session.send_client_content(
                        turns=contents, turn_complete=False
                    )
            history.advance_cursor()

    @override
    async def disconnect_live(self) -> None:
        if self._session_cm:
            await self._session_cm.__aexit__(None, None, None)
            self._session_cm = None
            self._session = None
        self._client = None
        self._history = None

    def _assert_session(self) -> "AsyncSession":
        if self._session is None:
            raise RuntimeError(
                "No active live session. Use 'async with agent.live()' to start a live session."
            )
        return self._session

    @override
    async def send_live_chunk(self, chunk: LiveChunk):
        session = self._assert_session()
        match chunk:
            case LiveChunkAudio(data=data, mime_type=mime):
                await session.send_realtime_input(
                    audio=types.Blob(data=data, mime_type=mime)
                )
            case LiveChunkVideo(data=data, mime_type=mime):
                await session.send_realtime_input(
                    video=types.Blob(data=data, mime_type=mime)
                )
            case LiveChunkImage(data=data, mime_type=mime):
                await session.send_realtime_input(
                    video=types.Blob(data=data, mime_type=mime)
                )
            case LiveChunkText(text=text):
                await session.send_realtime_input(text=text)
            case LiveChunkEnd():
                await session.send_realtime_input(activity_end={})

    @override
    async def send_tool_responses(self, responses: list[ToolCallResponse]) -> None:
        session = self._assert_session()
        function_responses = [
            types.FunctionResponse(
                id=r.tool_call_id,
                name=r.tool_name,
                response=(
                    r.output if isinstance(r.output, dict) else {"result": r.output}
                ),
            )
            for r in responses
        ]
        await session.send_tool_response(function_responses=function_responses)

    @override
    async def receive(self) -> AsyncGenerator[StreamPart, None]:
        session = self._assert_session()
        turn_started = False
        text_started = False
        text_id = str(uuid4())
        audio_started = False
        audio_id = str(uuid4())
        input_tx_started = False
        input_tx_id = str(uuid4())
        output_tx_started = False
        output_tx_id = str(uuid4())
        while True:
            async for response in session.receive():
                content = response.server_content
                if content:
                    if content.model_turn and content.model_turn.parts:
                        if not turn_started:
                            yield StreamPartTurnStart(role="assistant")
                            turn_started = True
                        for part in content.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                if not audio_started:
                                    yield StreamPartAudioStart(id=audio_id)
                                    audio_started = True
                                yield StreamPartAudioDelta(
                                    id=audio_id,
                                    delta=part.inline_data.data,
                                )
                            if part.text:
                                if not text_started:
                                    yield StreamPartTextStart(id=text_id)
                                    text_started = True
                                yield StreamPartTextDelta(id=text_id, delta=part.text)
                    if content.input_transcription and content.input_transcription.text:
                        if not input_tx_started:
                            yield StreamPartInputTranscriptionStart(id=input_tx_id)
                            input_tx_started = True
                        yield StreamPartInputTranscriptionDelta(
                            id=input_tx_id,
                            delta=content.input_transcription.text,
                        )
                    if (
                        content.output_transcription
                        and content.output_transcription.text
                    ):
                        if not output_tx_started:
                            yield StreamPartOutputTranscriptionStart(id=output_tx_id)
                            output_tx_started = True
                        yield StreamPartOutputTranscriptionDelta(
                            id=output_tx_id,
                            delta=content.output_transcription.text,
                        )
                    if content.interrupted is True:
                        if audio_started:
                            yield StreamPartAudioEnd(id=audio_id)
                            audio_started = False
                            audio_id = str(uuid4())
                        if text_started:
                            yield StreamPartTextEnd(id=text_id)
                            text_started = False
                            text_id = str(uuid4())
                        if output_tx_started:
                            yield StreamPartOutputTranscriptionEnd(id=output_tx_id)
                            output_tx_started = False
                            output_tx_id = str(uuid4())
                        yield StreamPartTurnEnd(
                            usage=_get_usage(response.usage_metadata),
                            finish_reason="interrupted",
                            role="assistant",
                            message=None,
                        )
                        turn_started = False
                    elif content.turn_complete or content.generation_complete:
                        if audio_started:
                            yield StreamPartAudioEnd(id=audio_id)
                            audio_started = False
                            audio_id = str(uuid4())
                        if text_started:
                            yield StreamPartTextEnd(id=text_id)
                            text_started = False
                            text_id = str(uuid4())
                        if input_tx_started:
                            yield StreamPartInputTranscriptionEnd(id=input_tx_id)
                            input_tx_started = False
                            input_tx_id = str(uuid4())
                        if output_tx_started:
                            yield StreamPartOutputTranscriptionEnd(id=output_tx_id)
                            output_tx_started = False
                            output_tx_id = str(uuid4())
                        yield StreamPartTurnEnd(
                            usage=_get_usage(response.usage_metadata),
                            finish_reason="stop",
                            role="assistant",
                            message=None,
                        )
                        turn_started = False

                tool_call_resp = response.tool_call
                if tool_call_resp and tool_call_resp.function_calls:
                    for fc in tool_call_resp.function_calls:
                        tc_id = fc.id or str(uuid4())
                        yield ToolCall(
                            tool_call_id=tc_id,
                            tool_name=fc.name or "",
                            input=dict(fc.args) if fc.args else {},
                        )

    # --- Request/response emulation over live session ---

    async def _send_new_messages(self, new_messages: list[Message]) -> None:
        """Send new messages that the live session hasn't seen yet."""
        if not new_messages:
            return
        session = self._assert_session()

        for msg in new_messages:
            # Send the message via realtime input (triggers model response)
            if isinstance(msg, UserMessage):
                await _send_user_message_realtime(session, msg)
            elif isinstance(msg, ToolMessage):
                # ToolMessage is user-originated content, send via realtime input
                for part in msg.content:
                    if isinstance(part, MessagePartToolResult):
                        response_data = (
                            part.output
                            if isinstance(part.output, dict)
                            else {"result": part.output}
                        )
                        await session.send_tool_response(
                            function_responses={
                                "id": part.tool_call_id,
                                "name": part.tool_name,
                                "response": response_data,
                            }
                        )
            else:
                raise NotImplementedError(
                    f"Unsupported message type for live input: {type(msg)}"
                )

    def _get_new_messages(self, messages: list[Message]) -> list[Message]:
        """Get messages the live session hasn't seen, using History cursor if available."""
        if self._history:
            new = self._history.get_new()
            self._history.advance_cursor()
            return list(new)
        # Fallback: treat all messages as new
        return messages

    @override
    async def generate(
        self,
        instructions: str,
        messages: list[Message],
        tools: ToolSet,
        options: LLMOptions,
        client: httpx.AsyncClient,
    ) -> GenerationResult:
        session = self._assert_session()

        await self._send_new_messages(self._get_new_messages(messages))
        await session.send_realtime_input(activity_end={})

        # Collect response
        text_parts: list[str] = []
        async for response in session.receive():
            content = response.server_content
            if content:
                if content.model_turn and content.model_turn.parts:
                    for part in content.model_turn.parts:
                        if part.text:
                            text_parts.append(part.text)
                if content.output_transcription and content.output_transcription.text:
                    text_parts.append(content.output_transcription.text)
                if (
                    content.turn_complete
                    or content.interrupted
                    or content.generation_complete
                ):
                    break

        full_text = "".join(text_parts)
        message = AssistantMessage(content=[MessagePartText(text=full_text)])
        return GenerationResult(message=message, finish_reason="stop", usage=Usage())

    @override
    async def stream(
        self,
        instructions: str,
        messages: list[Message],
        tools: ToolSet,
        options: LLMOptions,
        client: httpx.AsyncClient,
    ) -> AsyncGenerator[StreamPart, None]:
        session = self._assert_session()

        await self._send_new_messages(self._get_new_messages(messages))
        await session.send_realtime_input(activity_end={})

        text_started = False
        text_id = str(uuid4())
        started = False

        async for response in session.receive():
            if not started:
                started = True
                yield StreamPartTurnStart(role="assistant")

            content = response.server_content
            if content:
                if content.model_turn and content.model_turn.parts:
                    for part in content.model_turn.parts:
                        if part.text:
                            if not text_started:
                                yield StreamPartTextStart(id=text_id)
                                text_started = True
                            yield StreamPartTextDelta(id=text_id, delta=part.text)
                    if (
                        content.output_transcription
                        and content.output_transcription.text
                    ):
                        if not text_started:
                            yield StreamPartTextStart(id=text_id)
                            text_started = True
                        yield StreamPartTextDelta(
                            id=text_id, delta=content.output_transcription.text
                        )
                if (
                    content.turn_complete
                    or content.interrupted
                    or content.generation_complete
                ):
                    break

        if text_started:
            yield StreamPartTextEnd(id=text_id)

        yield StreamPartTurnEnd(
            usage=Usage(), finish_reason="stop", role="assistant", message=None
        )
