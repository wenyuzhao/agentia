import abc
import asyncio
from typing import AsyncGenerator, Literal, TYPE_CHECKING, Optional, Sequence, overload
from pydantic import BaseModel, Field
from agentia.llm.agentic import run_agent_loop_live
from agentia.spec import StreamPart
from typing import TYPE_CHECKING, Literal
from dataclasses import dataclass
from io import BytesIO
import threading

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


class EchoCanceller:
    """Acoustic echo canceller using frequency-domain adaptive filtering (FDAF/NLMS).

    Removes speaker output (reference) from microphone input to prevent echo feedback.
    Handles sample rate mismatch between input (16kHz) and output (24kHz) automatically.
    """

    def __init__(
        self,
        block_size: int = 256,
        num_blocks: int = 12,
        mu: float = 0.5,
        delta: float = 1e-2,
        input_rate: int = 16000,
        output_rate: int = 24000,
    ):
        import numpy as np

        self._np = np
        N = block_size
        M = num_blocks
        F = N + 1  # rfft output length for 2N-point FFT

        self._N = N
        self._M = M
        self._mu = mu
        self._delta = delta
        self._input_rate = input_rate
        self._output_rate = output_rate

        # Adaptive filter taps in frequency domain
        self._W = np.zeros((M, F), dtype=np.complex128)
        # Reference delay line in frequency domain
        self._X_buf = np.zeros((M, F), dtype=np.complex128)
        # Running power estimate for NLMS normalisation
        self._P = np.ones(F, dtype=np.float64) * delta

        # Overlap buffers (overlap-save needs previous block)
        self._ref_overlap = np.zeros(N, dtype=np.float64)
        self._mic_overlap = np.zeros(N, dtype=np.float64)

        # Queues for incomplete blocks
        self._ref_queue = np.array([], dtype=np.float64)
        self._mic_pending = np.array([], dtype=np.float64)
        self._out_pending = np.array([], dtype=np.float64)

        self._lock = threading.Lock()

    def _resample(self, data: "numpy.ndarray", from_rate: int, to_rate: int):  # type: ignore[name-defined]
        if from_rate == to_rate:
            return data
        np = self._np
        n_out = int(len(data) * to_rate / from_rate)
        x_old = np.arange(len(data))
        x_new = np.linspace(0, len(data) - 1, n_out)
        return np.interp(x_new, x_old, data)

    def feed_reference(self, data: bytes) -> None:
        """Feed playback audio as the reference signal. Called by AudioOutput."""
        np = self._np
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float64) / 32768.0
        if self._output_rate != self._input_rate:
            samples = self._resample(samples, self._output_rate, self._input_rate)
        with self._lock:
            self._ref_queue = np.concatenate([self._ref_queue, samples])

    def process(self, mic_data: bytes) -> bytes:
        """Remove echo from microphone audio. Called by AudioInput."""
        np = self._np
        N = self._N
        mic = np.frombuffer(mic_data, dtype=np.int16).astype(np.float64) / 32768.0
        n_input = len(mic)

        with self._lock:
            self._mic_pending = np.concatenate([self._mic_pending, mic])
            output_blocks: list = []

            while len(self._mic_pending) >= N:
                mic_block = self._mic_pending[:N]
                self._mic_pending = self._mic_pending[N:]

                # Get matching reference block (zero if not available yet)
                if len(self._ref_queue) >= N:
                    ref_block = self._ref_queue[:N]
                    self._ref_queue = self._ref_queue[N:]
                else:
                    ref_block = np.zeros(N, dtype=np.float64)
                    if len(self._ref_queue) > 0:
                        ref_block[: len(self._ref_queue)] = self._ref_queue
                        self._ref_queue = np.array([], dtype=np.float64)

                output_blocks.append(self._process_block(mic_block, ref_block))

            if output_blocks:
                self._out_pending = np.concatenate([self._out_pending, *output_blocks])

            # Return exactly the same number of samples as input
            if len(self._out_pending) >= n_input:
                result = self._out_pending[:n_input]
                self._out_pending = self._out_pending[n_input:]
            else:
                # Not enough processed samples yet – pass through raw input
                return mic_data

        result = np.clip(result * 32768.0, -32768, 32767).astype(np.int16)
        return result.tobytes()

    def _process_block(self, mic_block, ref_block):
        np = self._np
        N = self._N

        # Overlap-save: build 2N segments from previous + current block
        ref_seg = np.concatenate([self._ref_overlap, ref_block])
        mic_seg = np.concatenate([self._mic_overlap, mic_block])
        self._ref_overlap = ref_block.copy()
        self._mic_overlap = mic_block.copy()

        # Forward FFT
        X = np.fft.rfft(ref_seg)
        D = np.fft.rfft(mic_seg)

        # Shift reference delay line and insert newest block
        self._X_buf = np.roll(self._X_buf, 1, axis=0)
        self._X_buf[0] = X

        # Echo estimate: sum of filter * reference across all taps
        Y = np.sum(self._W * self._X_buf, axis=0)

        # Error = desired (mic) - echo estimate
        E = D - Y

        # Overlap-save: take last N samples from the 2N IFFT output
        e_time = np.fft.irfft(E, n=2 * N)
        out_block = e_time[N:]

        # Update running power estimate
        self._P = 0.9 * self._P + 0.1 * np.abs(X) ** 2

        # NLMS weight update
        step = self._mu * E / (self._P + self._delta)
        for k in range(self._M):
            self._W[k] += step * np.conj(self._X_buf[k])

        return out_block


@dataclass
class LiveChunkAudio:
    data: bytes
    mime_type: str = "audio/pcm;rate=16000"


@dataclass
class LiveChunkVideo:
    data: bytes
    mime_type: str = "image/jpeg"


@dataclass
class LiveChunkImage:
    data: bytes
    mime_type: str = "image/jpeg"


@dataclass
class LiveChunkText:
    text: str


@dataclass
class LiveChunkEnd: ...


LiveChunk = (
    LiveChunkAudio | LiveChunkVideo | LiveChunkImage | LiveChunkText | LiveChunkEnd
)


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

    async def send_audio(
        self, data: bytes, mime_type: str = "audio/pcm;rate=16000"
    ) -> None:
        """Send an audio chunk to the live session. Default: PCM 16kHz 16-bit mono."""
        if self.__multithread_mode:
            await self._input_queue.put(LiveChunkAudio(data=data, mime_type=mime_type))
            return
        await self.agent.provider.send_audio(data, mime_type)

    async def send_image(self, data: bytes, mime_type: str = "image/jpeg") -> None:
        """Send an image to the live session."""
        if self.__multithread_mode:
            await self._input_queue.put(LiveChunkImage(data=data, mime_type=mime_type))
            return
        await self.agent.provider.send_image(data, mime_type)

    async def send_video(self, data: bytes, mime_type: str = "image/jpeg") -> None:
        """Send a video frame to the live session."""
        if self.__multithread_mode:
            await self._input_queue.put(LiveChunkVideo(data=data, mime_type=mime_type))
            return
        await self.agent.provider.send_video(data, mime_type)

    async def send_text(self, text: str) -> None:
        """Send text input to the live session."""
        if self.__multithread_mode:
            await self._input_queue.put(LiveChunkText(text=text))
            return
        await self.agent.provider.send_text_live(text)

    async def send_audio_stream_end(self) -> None:
        """Signal end of audio stream to flush cached audio."""
        if self.__multithread_mode:
            await self._input_queue.put(LiveChunkEnd())
            return
        await self.agent.provider.send_audio_stream_end()

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
            match chunk:
                case LiveChunkText(text=text):
                    await self.agent.provider.send_text_live(text)
                case LiveChunkAudio(data=data, mime_type=mime_type):
                    await self.agent.provider.send_audio(data, mime_type)
                case LiveChunkImage(data=data, mime_type=mime_type):
                    await self.agent.provider.send_image(data, mime_type)
                case LiveChunkVideo(data=data, mime_type=mime_type):
                    await self.agent.provider.send_video(data, mime_type)
                case LiveChunkEnd():
                    await self.agent.provider.send_audio_stream_end()

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
