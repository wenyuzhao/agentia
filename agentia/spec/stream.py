from datetime import datetime
from typing import Annotated, Literal, Sequence
from pydantic import BaseModel, Field
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


class StreamPartStreamStart(BaseModel):
    type: Literal["stream-start"] = "stream-start"
    id: str | None = None
    timestamp: datetime | None = None
    model_id: str | None = None


class StreamPartStreamEnd(BaseModel):
    type: Literal["finish"] = "finish"
    usage: Usage
    finish_reason: FinishReason


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
    | StreamPartStreamStart
    | StreamPartStreamEnd,
    Field(discriminator="type"),
]

__all__ = [
    "StreamPartTextStart",
    "StreamPartTextDelta",
    "StreamPartTextEnd",
    "StreamPartReasoningStart",
    "StreamPartReasoningDelta",
    "StreamPartReasoningEnd",
    "StreamPartStreamStart",
    "StreamPartStreamEnd",
    "StreamPart",
]
