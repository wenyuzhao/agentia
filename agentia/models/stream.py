from typing import Annotated, Literal, Sequence
from pydantic import BaseModel, Field

from agentia.models.chat import AssistantMessage, ToolMessage
from .base import *


class StreamPartTextStart(BaseModel):
    type: Literal["text-start"] = "text-start"
    id: str
    provider_metadata: ProviderMetadata | None = None


class StreamPartTextDelta(BaseModel):
    type: Literal["text-delta"] = "text-delta"
    delta: str
    id: str
    provider_metadata: ProviderMetadata | None = None


class StreamPartTextEnd(BaseModel):
    type: Literal["text-end"] = "text-end"
    id: str
    provider_metadata: ProviderMetadata | None = None


class StreamPartReasoningStart(BaseModel):
    type: Literal["reasoning-start"] = "reasoning-start"
    id: str
    provider_metadata: ProviderMetadata | None = None


class StreamPartReasoningDelta(BaseModel):
    type: Literal["reasoning-delta"] = "reasoning-delta"
    delta: str
    id: str
    provider_metadata: ProviderMetadata | None = None


class StreamPartReasoningEnd(BaseModel):
    type: Literal["reasoning-end"] = "reasoning-end"
    id: str
    provider_metadata: ProviderMetadata | None = None


class StreamPartAudioStart(BaseModel):
    type: Literal["audio-start"] = "audio-start"
    id: str


class StreamPartAudioDelta(BaseModel):
    type: Literal["audio-delta"] = "audio-delta"
    id: str
    delta: bytes
    """PCM audio data. Output is 24kHz, 16-bit, mono, little-endian."""


class StreamPartAudioEnd(BaseModel):
    type: Literal["audio-end"] = "audio-end"
    id: str


class StreamPartInputTranscriptionStart(BaseModel):
    type: Literal["input-transcription-start"] = "input-transcription-start"
    id: str


class StreamPartInputTranscriptionDelta(BaseModel):
    type: Literal["input-transcription-delta"] = "input-transcription-delta"
    id: str
    delta: str


class StreamPartInputTranscriptionEnd(BaseModel):
    type: Literal["input-transcription-end"] = "input-transcription-end"
    id: str


class StreamPartOutputTranscriptionStart(BaseModel):
    type: Literal["output-transcription-start"] = "output-transcription-start"
    id: str


class StreamPartOutputTranscriptionDelta(BaseModel):
    type: Literal["output-transcription-delta"] = "output-transcription-delta"
    id: str
    delta: str


class StreamPartOutputTranscriptionEnd(BaseModel):
    type: Literal["output-transcription-end"] = "output-transcription-end"
    id: str


class StreamPartTurnStart(BaseModel):
    type: Literal["turn-start"] = "turn-start"
    role: Literal["assistant", "tool"]


class StreamPartTurnEnd(BaseModel):
    type: Literal["turn-end"] = "turn-end"
    usage: Usage = Usage()
    finish_reason: FinishReason = "stop"
    role: Literal["assistant", "tool"]
    message: Annotated[
        AssistantMessage | ToolMessage | None, Field(discriminator="role")
    ]


class StreamPartStart(BaseModel):
    type: Literal["start"] = "start"


class StreamPartEnd(BaseModel):
    type: Literal["end"] = "end"
    usage: Usage = Usage()
    finish_reason: FinishReason = "stop"


class Annotations(BaseModel):
    type: Literal["annotations"] = "annotations"
    annotations: Sequence[Annotation]

    def __init__(
        self,
        annotations: Sequence[Annotation],
        *,
        type: Literal["annotations"] = "annotations",
    ):
        super().__init__(type=type, annotations=annotations)


type StreamPart = Annotated[
    StreamPartTextStart
    | StreamPartTextDelta
    | StreamPartTextEnd
    | StreamPartReasoningStart
    | StreamPartReasoningDelta
    | StreamPartReasoningEnd
    | ToolCall
    | ToolCallResponse
    | Annotation
    | StreamPartAudioStart
    | StreamPartAudioDelta
    | StreamPartAudioEnd
    | StreamPartInputTranscriptionStart
    | StreamPartInputTranscriptionDelta
    | StreamPartInputTranscriptionEnd
    | StreamPartOutputTranscriptionStart
    | StreamPartOutputTranscriptionDelta
    | StreamPartOutputTranscriptionEnd
    | StreamPartTurnStart
    | StreamPartTurnEnd
    | StreamPartStart
    | StreamPartEnd,
    Field(discriminator="type"),
]

__all__ = [
    "StreamPartTextStart",
    "StreamPartTextDelta",
    "StreamPartTextEnd",
    "StreamPartReasoningStart",
    "StreamPartReasoningDelta",
    "StreamPartReasoningEnd",
    "StreamPartAudioStart",
    "StreamPartAudioDelta",
    "StreamPartAudioEnd",
    "StreamPartInputTranscriptionStart",
    "StreamPartInputTranscriptionDelta",
    "StreamPartInputTranscriptionEnd",
    "StreamPartOutputTranscriptionStart",
    "StreamPartOutputTranscriptionDelta",
    "StreamPartOutputTranscriptionEnd",
    "StreamPartTurnStart",
    "StreamPartTurnEnd",
    "StreamPartStart",
    "StreamPartEnd",
    "StreamPart",
]
