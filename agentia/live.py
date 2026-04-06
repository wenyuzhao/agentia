import abc
import asyncio
from typing import AsyncGenerator, Literal, TYPE_CHECKING, Optional, Sequence, overload
from pydantic import BaseModel, Field
from agentia.llm.agentic import run_agent_loop_live
from agentia.spec import StreamPart
from typing import TYPE_CHECKING, Literal
from io import BytesIO
from agentia.utils.aec import EchoCanceller
from agentia.spec.live import (
    LiveChunk,
    LiveChunkAudio,
    LiveChunkText,
    LiveChunkImage,
    LiveChunkVideo,
    LiveChunkEnd,
)

if TYPE_CHECKING:
    from agentia.agent import Agent


class LiveOptions(BaseModel):
    """Configuration for a Gemini Live session."""

    modalities: list[Literal["text", "audio"]] = Field(
        default_factory=lambda: ["audio"]
    )
    """Response modalities. Currently only AUDIO is reliably supported by Gemini Live API."""

    voice: str | None = None
    """Voice name (e.g. "Puck", "Charon", "Kore", "Fenrir", "Aoede")."""

    language: str | None = None
    """Language code (e.g. "en")."""

    thinking_level: Literal["minimal", "low", "medium", "high"] | None = None
    """Thinking level for native audio models. Default is "minimal" for lowest latency."""

    vad: bool = True
    """Enable voice activity detection for automatic interruption handling."""

    aec: bool = False
    """Enable acoustic echo cancellation to suppress speaker-to-mic feedback."""


class InputStream(abc.ABC):
    def __init__(self):
        self._live: Optional["Live"] = None

    async def init(self): ...
    async def deinit(self): ...

    async def send(self, chunk: LiveChunk) -> None:
        """Send a chunk to the input stream."""
        assert self._live is not None, "InputStream not connected to Live session"
        await self._live._input_queue.put(chunk)

    @abc.abstractmethod
    async def run(self): ...


class OutputStream(abc.ABC):
    def __init__(self):
        self._live: Optional["Live"] = None

    async def init(self): ...
    async def deinit(self): ...

    @overload
    async def receive(self, kind: Literal["audio"]) -> LiveChunkAudio | None: ...

    @overload
    async def receive(
        self, kind: Literal["text", "transcription"]
    ) -> LiveChunkText | None: ...

    async def receive(
        self, kind: Literal["text", "audio", "transcription"]
    ) -> LiveChunkAudio | LiveChunkText | None:
        """Receive a chunk from the output stream."""
        assert self._live is not None, "OutputStream not connected to Live session"
        if kind == "text":
            return await self._live._output_text_queue.get()
        elif kind == "audio":
            return await self._live._output_audio_queue.get()
        elif kind == "transcription":
            return await self._live._output_transcription_queue.get()

    @abc.abstractmethod
    async def run(self): ...


class VideoInput(InputStream):
    def __init__(
        self,
        kind: Literal["camera", "screen"],
        device: int | None = None,
        fps: float = 1,
    ):
        super().__init__()
        self.kind = kind
        self.fps = fps
        self.device = device
        self.cap = None

    async def init(self):
        if self.kind == "camera":
            import cv2

            self.cap = cv2.VideoCapture(self.device or 0)
            self.grab_camera_frame()
        else:
            self.grab_screenshot()

    async def deinit(self):
        if self.cap is not None:
            self.cap.release()

    def grab_screenshot(self) -> bytes:
        import pyscreenshot

        img = pyscreenshot.grab()
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG")  # type: ignore
        return buf.getvalue()

    def grab_camera_frame(self) -> bytes:
        import cv2
        from PIL import Image

        assert self.cap is not None, "Camera capture not initialized"
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to grab camera frame")
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        img.save(buf, format="JPEG")  # type: ignore
        return buf.getvalue()

    async def run(self):
        while True:
            await asyncio.sleep(1.0 / float(self.fps))
            if self.kind == "screen":
                data = await asyncio.to_thread(self.grab_screenshot)
            else:
                data = await asyncio.to_thread(self.grab_camera_frame)
            await self.send(LiveChunkVideo(data=data, mime_type="image/jpeg"))


class AudioInput(InputStream):
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        device: int | None = None,
        echo_canceller: EchoCanceller | None = None,
    ):
        super().__init__()

        import pyaudio

        self.pya = pyaudio.PyAudio()
        self.format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.stream = None
        self.device = device
        self.echo_canceller = echo_canceller

    async def init(self):
        self.stream = await asyncio.to_thread(
            self.pya.open,
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=self.device,
        )

    async def deinit(self):
        if self.stream is not None:
            self.stream.close()
        self.pya.terminate()

    async def run(self):
        assert self.stream is not None, "Audio stream not initialized"
        while True:
            data = await asyncio.to_thread(
                self.stream.read, self.chunk_size, exception_on_overflow=False
            )
            if self.echo_canceller is not None:
                data = await asyncio.to_thread(self.echo_canceller.process, data)
            await self.send(LiveChunkAudio(data=data, mime_type="audio/pcm;rate=16000"))


class AudioOutput(OutputStream):
    def __init__(
        self,
        sample_rate: int = 24000,
        device: int | None = None,
        echo_canceller: EchoCanceller | None = None,
    ):
        super().__init__()

        import pyaudio

        self.pya = pyaudio.PyAudio()
        self.format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = sample_rate
        self.stream = None
        self.device = device
        self.echo_canceller = echo_canceller

    async def init(self):
        self.stream = await asyncio.to_thread(
            self.pya.open,
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            output_device_index=self.device,
        )

    async def deinit(self):
        if self.stream is not None:
            self.stream.close()
        self.pya.terminate()

    async def run(self):
        assert self.stream is not None, "Audio stream not initialized"
        while True:
            chunk = await self.receive("audio")
            if chunk is None:
                continue
            if self.echo_canceller is not None:
                self.echo_canceller.feed_reference(chunk.data)
            await asyncio.to_thread(self.stream.write, chunk.data)


class TextOutput(OutputStream):
    def __init__(self):
        super().__init__()

    async def run(self):
        while True:
            chunk = await self.receive("text")
            if chunk is None:
                print()
            else:
                print(chunk.text, end="", flush=True)


class TranscriptionOutput(OutputStream):
    def __init__(self):
        super().__init__()

    async def run(self):
        while True:
            chunk = await self.receive("transcription")
            if chunk is None:
                print()
            else:
                print(chunk.text, end="", flush=True)


class Live:
    def __init__(self, agent: "Agent", options: LiveOptions | None = None):
        self.agent = agent
        self.options = options or LiveOptions()
        self.__nesting_level = 0
        self.__multithread_mode = False
        self._input_queue = asyncio.Queue[LiveChunk]()
        self._output_audio_queue = asyncio.Queue[LiveChunkAudio | None]()
        self._output_text_queue = asyncio.Queue[LiveChunkText | None]()
        self._output_transcription_queue = asyncio.Queue[LiveChunkText | None]()

    async def __aenter__(self):
        self.__nesting_level += 1
        if self.__nesting_level > 1:
            # Already in a live session, just return self without reconnecting
            return self
        await self.agent.__aenter__()
        # Connect live session if provider supports it
        assert self.agent.provider.supports_live
        await self.agent.tools.init()
        instructions = self.agent.history.get_instructions() or None
        await self.agent.provider.connect_live(
            self.options, self.agent.tools, instructions, history=self.agent.history
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.__nesting_level -= 1
        if self.__nesting_level > 0:
            # Still in a nested live session, do not disconnect yet
            return
        # Disconnect live session if provider supports it
        await self.agent.provider.disconnect_live()
        await self.agent.__aexit__(exc_type, exc_val, exc_tb)

    @overload
    async def send(self, *, chunk: LiveChunk): ...
    @overload
    async def send(self, *, text: str): ...
    @overload
    async def send(self, *, audio: bytes, mime_type: str | None = None): ...
    @overload
    async def send(self, *, image: bytes, mime_type: str | None = None): ...
    @overload
    async def send(self, *, video: bytes, mime_type: str | None = None): ...
    @overload
    async def send(self, *, end: Literal[True]): ...

    async def send(
        self,
        *,
        image: bytes | None = None,
        video: bytes | None = None,
        audio: bytes | None = None,
        text: str | None = None,
        end: Literal[True] | None = None,
        mime_type: Optional[str] = None,
        chunk: LiveChunk | None = None,
    ):
        """Send a chunk to the live session."""
        args = dict(
            image=image, video=video, audio=audio, text=text, end=end, chunk=chunk
        )

        def check_args(active: str):
            for k, v in args.items():
                if k != active:
                    assert v is None, f"`{k}` should not be provided"
            if active in ("text", "end", "chunk"):
                assert mime_type is None, "mime_type should not be specified"

        if chunk:
            check_args("chunk")
        else:
            if image:
                check_args("image")
                chunk = LiveChunkImage(data=image, mime_type=mime_type or "image/jpeg")
            elif video:
                check_args("video")
                chunk = LiveChunkVideo(data=video, mime_type=mime_type or "image/jpeg")
            elif audio:
                check_args("audio")
                chunk = LiveChunkAudio(
                    data=audio, mime_type=mime_type or "audio/pcm;rate=16000"
                )
            elif text:
                check_args("text")
                chunk = LiveChunkText(text=text)
            elif end is True:
                check_args("end")
                chunk = LiveChunkEnd()
            else:
                raise ValueError("Invalid input: no data provided")

        if self.__multithread_mode:
            await self._input_queue.put(chunk)
            return
        await self.agent.provider.send_live_chunk(chunk)

    async def receive(self) -> AsyncGenerator[StreamPart, None]:
        """Receive stream parts from the live session."""
        if self.__multithread_mode:
            raise RuntimeError(
                "Multithread mode does not support receive() - use start() instead"
            )
        async for event in run_agent_loop_live(self.agent):
            yield event

    async def __receive_task(self):
        async for event in run_agent_loop_live(self.agent):
            match event.type:
                case "text-delta":
                    chunk = LiveChunkText(text=event.delta)
                    await self._output_text_queue.put(chunk)
                case "text-end":
                    await self._output_text_queue.put(None)
                case "audio-delta":
                    chunk = LiveChunkAudio(
                        data=event.delta, mime_type="audio/pcm;rate=16000"
                    )
                    await self._output_audio_queue.put(chunk)
                case "audio-end":
                    await self._output_audio_queue.put(None)
                case "output-transcription-delta":
                    chunk = LiveChunkText(text=event.delta)
                    await self._output_transcription_queue.put(chunk)
                case "output-transcription-end":
                    await self._output_transcription_queue.put(None)

    async def __send_task(self):
        while True:
            chunk = await self._input_queue.get()
            await self.agent.provider.send_live_chunk(chunk)

    async def start(
        self,
        inputs: (
            Sequence[InputStream | Literal["audio", "camera", "screen"]] | None
        ) = None,
        outputs: (
            Sequence[OutputStream | Literal["text", "audio", "transcription"]] | None
        ) = None,
        camera: Optional[int] = None,
        screen: Optional[int] = None,
        mic: Optional[int] = None,
        speaker: Optional[int] = None,
    ):
        """Start the live session in multithread mode, allowing concurrent send/receive."""
        self.__multithread_mode = True

        # Create shared echo canceller if AEC is enabled
        aec = self.options.aec
        echo_canceller: EchoCanceller | None = None

        def _get_echo_canceller() -> EchoCanceller:
            nonlocal echo_canceller
            if echo_canceller is None:
                echo_canceller = EchoCanceller()
            return echo_canceller

        def create_istream(kind: Literal["audio", "camera", "screen"]) -> InputStream:
            if kind == "audio":
                return AudioInput(
                    device=mic,
                    echo_canceller=_get_echo_canceller() if aec else None,
                )
            elif kind == "camera":
                return VideoInput(kind="camera", device=camera)
            elif kind == "screen":
                return VideoInput(kind="screen", device=screen)
            else:
                raise NotImplementedError(
                    f"Input stream for {kind} not implemented yet"
                )

        def create_ostream(
            kind: Literal["text", "audio", "transcription"],
        ) -> OutputStream:
            if kind == "audio":
                return AudioOutput(
                    device=speaker,
                    echo_canceller=_get_echo_canceller() if aec else None,
                )
            elif kind == "text":
                return TextOutput()
            elif kind == "transcription":
                return TranscriptionOutput()
            else:
                raise NotImplementedError(
                    f"Output stream for {kind} not implemented yet"
                )

        inputs = [
            i if isinstance(i, InputStream) else create_istream(i) for i in inputs or []
        ]
        outputs = [
            o if isinstance(o, OutputStream) else create_ostream(o)
            for o in outputs or []
        ]

        for o in outputs:
            await o.init()
        for i in inputs:
            await i.init()

        try:
            async with self:
                async with asyncio.TaskGroup() as tg:
                    tasks = []
                    for i in inputs:
                        i._live = self
                        tasks.append(i.run())
                    for o in outputs:
                        o._live = self
                        tasks.append(o.run())
                    for t in tasks:
                        tg.create_task(t)
                    tg.create_task(self.__receive_task())
                    tg.create_task(self.__send_task())
        finally:
            for o in outputs:
                await o.deinit()
            for i in inputs:
                await i.deinit()


__all__ = ["LiveOptions", "Live", "EchoCanceller"]
