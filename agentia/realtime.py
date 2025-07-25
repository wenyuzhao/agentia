import asyncio
from contextlib import AsyncExitStack, ExitStack
from typing import AsyncGenerator, override
from typing_extensions import Literal
from agentia.agent import Agent
from agentia.llm import LLMBackend
from agentia.llm.google import GoogleBackend
from google.genai.types import (
    Modality,
    FunctionDeclaration,
    Tool,
    LiveConnectConfig,
    Behavior,
    FunctionResponseScheduling,
    ContentDict,
    Content,
    Blob,
    ContextWindowCompressionConfig,
    SlidingWindow,
)
import pyaudio

from agentia.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
    is_message,
)
import PIL.Image
from PIL.Image import Image
import abc
from mss import mss

CHUNK = 4200
FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 24000


class InputStream(abc.ABC):
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self.session: "RealtimeSession"

    @abc.abstractmethod
    async def start(self): ...

    async def stop(self):
        assert self._task
        self._task.cancel()


class ScreenRecording(InputStream):
    def __init__(self, monitor: int = 1, frame_rate: int = 4) -> None:
        super().__init__()
        self.frame_rate = frame_rate
        self.exit_stack = ExitStack()
        self.sct = self.exit_stack.enter_context(mss())
        self.monitor = monitor

    @override
    async def start(self):
        print("Capturing screen...")
        try:
            while True:
                sct_img = self.sct.grab(self.sct.monitors[self.monitor])
                img = PIL.Image.frombytes(
                    "RGB", sct_img.size, sct_img.bgra, "raw", "BGRX"
                )
                await self.session.send(img)
                await asyncio.sleep(1.0 / float(self.frame_rate))
        except asyncio.CancelledError:
            return

    @override
    async def stop(self):
        await super().stop()
        self.exit_stack.close()


class Microphone(InputStream):
    def __init__(self) -> None:
        super().__init__()

        self.p = pyaudio.PyAudio()

    @override
    async def start(self):
        print("Capturing microphone...")
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        try:
            while True:
                frame = self.stream.read(CHUNK)
                await self.session.send(Blob(data=frame, mime_type="audio/pcm"))
                await asyncio.sleep(10**-12)
        except asyncio.CancelledError:
            return
        finally:
            self.stream.close()


class RealtimeSession:
    """
    Only support Gimini models for now.
    The recommended model is `[google]gemini-2.5-flash-preview-native-audio-dialog`
    """

    def __init__(
        self,
        agent: Agent,
        backend: LLMBackend,
        response_modality: Literal["text", "audio"],
    ):
        self.agent = agent
        assert isinstance(
            backend, GoogleBackend
        ), "Only Gemini is supported for realtime sessions."
        assert not backend.model.endswith(":think"), "Reasoning is not allowed"
        self.llm: GoogleBackend = backend
        self.exit_stack = AsyncExitStack()
        self._tool_scheduling: Literal["blocking", "idle", "silent", "interrupt"] = (
            "idle"
        )
        self.response_modality = response_modality
        if self.response_modality == "audio":
            self.p: pyaudio.PyAudio | None = pyaudio.PyAudio()

    async def __aenter__(self):
        await self.agent.init()
        self.session = await self.exit_stack.enter_async_context(
            self.llm.client.aio.live.connect(
                model=self.llm.model,
                config=LiveConnectConfig(
                    response_modalities=[
                        (
                            Modality.TEXT
                            if self.response_modality == "text"
                            else Modality.AUDIO
                        )
                    ],
                    system_instruction=self.llm.history.instructions,
                    context_window_compression=(
                        ContextWindowCompressionConfig(sliding_window=SlidingWindow())
                    ),
                    tools=[
                        Tool(
                            function_declarations=[
                                FunctionDeclaration(
                                    behavior=(
                                        Behavior.BLOCKING
                                        if self._tool_scheduling == "blocking"
                                        else Behavior.NON_BLOCKING
                                    ),
                                    **s["function"],
                                )
                                for s in self.llm.tools.get_schema()
                            ]
                        ),
                    ],
                    # enable_affective_dialog=(
                    #     True if self.response_modality == "audio" else None
                    # ),
                ),
            )
        )
        contents: list[Content | ContentDict] = [
            self.llm.message_to_genai_content(m)
            for m in self.llm.history.messages
            if is_message(m)
        ]
        if contents:
            await self.session.send_client_content(turns=contents, turn_complete=False)
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.exit_stack.aclose()

    def stream(self, input: InputStream) -> None:
        """
        Send a stream of input data to the session.
        This is used to send user input in real-time.
        """
        input.session = self
        task = asyncio.create_task(input.start())
        input._task = task

    async def send(self, input: str | Image | Blob):
        """
        Send a message to the session.
        This is used to send commands or messages to the agent in real-time.
        """
        if isinstance(input, Image):
            await self.session.send_realtime_input(media=input)
        elif isinstance(input, Blob):
            await self.session.send_realtime_input(media=input)
        else:
            await self.session.send_realtime_input(text=input)

    async def __process_tool_calls(self, tool_calls: list[ToolCall]):
        if tool_calls:
            responses: list[ToolMessage] = []
            async for event in self.llm.tools.call_tools(tool_calls):
                if isinstance(event, ToolMessage):
                    responses.append(event)
                    self.llm.history.add(event)
                else:
                    # assert is_event(event), "Event must be a Event object"
                    # if events:
                    #     self.history.add(event)
                    #     yield event
                    ...
            print(responses)
            sched: FunctionResponseScheduling | None = None
            match self._tool_scheduling:
                case "blocking":
                    sched = None
                case "idle":
                    sched = FunctionResponseScheduling.WHEN_IDLE
                case "silent":
                    sched = FunctionResponseScheduling.SILENT
                case "interrupt":
                    sched = FunctionResponseScheduling.INTERRUPT
            await self.session.send_tool_response(
                function_responses=[
                    GoogleBackend.tool_message_to_genai_tool_response(
                        m, scheduling=sched
                    )
                    for m in responses
                ]
            )
            return responses
        return []

    async def receive(self) -> AsyncGenerator[str, None]:
        """
        Receive messages from the session.
        This is a generator that yields messages as they are received.
        """
        output_stream: pyaudio.Stream | None = None
        if self.response_modality == "audio":
            assert self.p is not None, "PyAudio is not initialized"
            output_stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=OUTPUT_RATE,
                output=True,
                frames_per_buffer=CHUNK,
            )
        while True:
            input_transcript = ""
            output_msg = AssistantMessage(content="")
            async for m in self.session.receive():
                if m.server_content:
                    # Update input transcript and add to history
                    if m.server_content.input_transcription:
                        if m.server_content.input_transcription.text:
                            input_transcript += (
                                m.server_content.input_transcription.text
                            )
                        if m.server_content.input_transcription.finished:
                            if input_transcript:
                                self.llm.history.messages.append(
                                    UserMessage(content=input_transcript)
                                )
                            input_transcript = ""
                    # Update output transcript
                    if m.server_content.output_transcription:
                        if m.server_content.output_transcription.text:
                            output_msg.content += (
                                m.server_content.output_transcription.text
                            )
                if m.tool_call and m.tool_call.function_calls:
                    print(m.tool_call.function_calls)
                    tool_calls = [
                        GoogleBackend.genai_tool_call_to_tool_call(t)
                        for t in m.tool_call.function_calls
                    ]
                    output_msg.tool_calls = tool_calls
                    self.llm.history.add(output_msg)
                    output_msg = AssistantMessage(content="")
                    asyncio.create_task(
                        self.__process_tool_calls(tool_calls=tool_calls)
                    )

                if m.text:
                    output_msg.content += m.text
                    yield m.text
                if m.server_content and m.server_content.model_turn:
                    for part in m.server_content.model_turn.parts or []:
                        if part.inline_data and part.inline_data.data:
                            audio_data = part.inline_data.data
                            assert output_stream is not None
                            output_stream.write(audio_data)
                            # await asyncio.sleep(10**-12)
            if output_msg.content or output_msg.tool_calls:
                self.llm.history.add(output_msg)
            # A turn is finished
            await asyncio.sleep(0.1)
