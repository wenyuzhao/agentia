from typing import Annotated, Literal, Sequence
from pydantic import BaseModel, Field

from agentia.models.chat import AssistantMessage, ToolMessage
from .base import *


class StreamPartTextStart(BaseModel):
    type: Literal["text-start"] = "text-start"
    id: str


class StreamPartTextDelta(BaseModel):
    type: Literal["text-delta"] = "text-delta"
    delta: str
    id: str


class StreamPartTextEnd(BaseModel):
    type: Literal["text-end"] = "text-end"
    id: str


class StreamPartReasoningStart(BaseModel):
    type: Literal["reasoning-start"] = "reasoning-start"
    id: str


class StreamPartReasoningDelta(BaseModel):
    type: Literal["reasoning-delta"] = "reasoning-delta"
    delta: str
    id: str


class StreamPartReasoningEnd(BaseModel):
    type: Literal["reasoning-end"] = "reasoning-end"
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
    "StreamPartTurnStart",
    "StreamPartTurnEnd",
    "StreamPartStart",
    "StreamPartEnd",
    "StreamPart",
]
