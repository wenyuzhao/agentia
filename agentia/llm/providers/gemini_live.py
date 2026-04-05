import os
from typing import TYPE_CHECKING, Any, AsyncGenerator, override
from uuid import uuid4

import httpx
from google import genai
from google.genai import types

if TYPE_CHECKING:
    from google.genai.live import AsyncSession

from agentia.live import LiveOptions
from agentia.llm import LLMOptions
from agentia.llm.providers import GenerationResult, Provider
from agentia.spec.base import FunctionTool, Usage
from agentia.spec.chat import (
    AssistantMessage,
    Message,
    MessagePartText,
)
from agentia.spec.base import ToolCall
from agentia.spec.stream import (
    StreamPart,
    StreamPartAudioDelta,
    StreamPartAudioEnd,
    StreamPartAudioStart,
    StreamPartMessageEnd,
    StreamPartMessageStart,
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

    input_audio_transcription: types.AudioTranscriptionConfig | None = None
    if options.enable_input_transcription:
        input_audio_transcription = types.AudioTranscriptionConfig()

    output_audio_transcription: types.AudioTranscriptionConfig | None = None
    if options.enable_output_transcription:
        output_audio_transcription = types.AudioTranscriptionConfig()

    context_window_compression: types.ContextWindowCompressionConfig | None = None
    if options.context_window_compression:
        context_window_compression = types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow()
        )

    session_resumption: types.SessionResumptionConfig | None = None
    if options.session_resumption:
        session_resumption = types.SessionResumptionConfig(transparent=True)

    realtime_input_config: types.RealtimeInputConfig | None = None
    if not options.vad_enabled:
        realtime_input_config = types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=True
            )
        )

    genai_tools: Any = _convert_tools_to_genai(tools)

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
    )


class GeminiLive(Provider):
    name = "gemini-live"

    def __init__(self, model: str):
        super().__init__(name="gemini-live", model=model)
        self._session: "AsyncSession | None" = None
        self._session_cm: Any = None  # The async context manager from connect()
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY or GEMINI_API_KEY environment variable must be set"
            )
        return genai.Client(api_key=api_key)

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
    ) -> None:
        self._client = self._get_client()
        config = _build_config(options, tools, instructions)
        self._session_cm = self._client.aio.live.connect(
            model=self.model, config=config
        )
        self._session = await self._session_cm.__aenter__()

    @override
    async def disconnect_live(self) -> None:
        if self._session_cm:
            await self._session_cm.__aexit__(None, None, None)
            self._session_cm = None
            self._session = None
        self._client = None

    def _assert_session(self) -> "AsyncSession":
        if self._session is None:
            raise RuntimeError(
                "No active live session. Use 'async with agent:' to start a session."
            )
        return self._session

    @override
    async def send_audio(
        self, data: bytes, mime_type: str = "audio/pcm;rate=16000"
    ) -> None:
        session = self._assert_session()
        await session.send_realtime_input(
            audio=types.Blob(data=data, mime_type=mime_type)
        )

    @override
    async def send_video(
        self, data: bytes, mime_type: str = "image/jpeg"
    ) -> None:
        session = self._assert_session()
        await session.send_realtime_input(
            video=types.Blob(data=data, mime_type=mime_type)
        )

    @override
    async def send_text_live(self, text: str) -> None:
        session = self._assert_session()
        await session.send_realtime_input(text=text)

    @override
    async def send_audio_stream_end(self) -> None:
        session = self._assert_session()
        await session.send_realtime_input(audio_stream_end=True)

    @override
    async def send_tool_response(self, tool_call_id: str, output: object) -> None:
        session = self._assert_session()
        response_data = output if isinstance(output, dict) else {"result": output}
        await session.send_tool_response(
            function_responses=[
                types.FunctionResponse(id=tool_call_id, response=response_data)
            ]
        )

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
                            yield StreamPartTurnStart()
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
                    if content.output_transcription and content.output_transcription.text:
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
                        )
                        turn_started = False
                    if content.turn_complete is True:
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

    @override
    async def generate(
        self,
        messages: list[Message],
        tools: ToolSet,
        options: LLMOptions,
        client: httpx.AsyncClient,
    ) -> GenerationResult:
        session = self._assert_session()
        # Send the last user message as text
        last_text = ""
        for msg in reversed(messages):
            if msg.role == "user":
                if isinstance(msg.content, str):
                    last_text = msg.content
                else:
                    parts = []
                    for p in msg.content:
                        if isinstance(p, MessagePartText):
                            parts.append(p.text)
                    last_text = " ".join(parts)
                break
            elif msg.role == "system":
                last_text = msg.content
                break

        if last_text:
            await session.send_realtime_input(text=last_text)

        # Collect response
        text_parts: list[str] = []
        async for response in session.receive():
            content = response.server_content
            if content:
                if content.model_turn and content.model_turn.parts:
                    for part in content.model_turn.parts:
                        if part.text:
                            text_parts.append(part.text)
                if content.turn_complete is True:
                    break

        full_text = "".join(text_parts)
        message = AssistantMessage(content=[MessagePartText(text=full_text)])
        return GenerationResult(
            message=message,
            finish_reason="stop",
            usage=Usage(),
            provider_metadata=None,
        )

    @override
    async def stream(
        self,
        messages: list[Message],
        tools: ToolSet,
        options: LLMOptions,
        client: httpx.AsyncClient,
    ) -> AsyncGenerator[StreamPart, None]:
        session = self._assert_session()
        # Send the last user message as text
        last_text = ""
        for msg in reversed(messages):
            if msg.role == "user":
                if isinstance(msg.content, str):
                    last_text = msg.content
                else:
                    parts = []
                    for p in msg.content:
                        if isinstance(p, MessagePartText):
                            parts.append(p.text)
                    last_text = " ".join(parts)
                break
            elif msg.role == "system":
                last_text = msg.content
                break

        yield StreamPartTurnStart()
        yield StreamPartMessageStart(role="assistant")

        if last_text:
            await session.send_realtime_input(text=last_text)

        text_started = False
        text_id = str(uuid4())

        async for response in session.receive():
            content = response.server_content
            if content:
                if content.model_turn and content.model_turn.parts:
                    for part in content.model_turn.parts:
                        if part.text:
                            if not text_started:
                                yield StreamPartTextStart(id=text_id)
                                text_started = True
                            yield StreamPartTextDelta(id=text_id, delta=part.text)
                if content.turn_complete is True:
                    break

        if text_started:
            yield StreamPartTextEnd(id=text_id)

        yield StreamPartMessageEnd(
            role="assistant",
            message=AssistantMessage(content=[MessagePartText(text="")]),
        )
        yield StreamPartTurnEnd(usage=Usage(), finish_reason="stop")
