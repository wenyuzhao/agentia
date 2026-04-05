from typing import Annotated, Literal
from pydantic import BaseModel, Field, JsonValue


class LiveEventAudio(BaseModel):
    type: Literal["audio"] = "audio"
    data: bytes
    """PCM audio data. Output is 24kHz, 16-bit, mono, little-endian."""


class LiveEventText(BaseModel):
    type: Literal["text"] = "text"
    text: str
    """Text response from the model."""


class LiveEventInputTranscription(BaseModel):
    type: Literal["input_transcription"] = "input_transcription"
    text: str
    """Transcription of user audio input."""


class LiveEventOutputTranscription(BaseModel):
    type: Literal["output_transcription"] = "output_transcription"
    text: str
    """Transcription of model audio output."""


class LiveEventInterrupted(BaseModel):
    type: Literal["interrupted"] = "interrupted"


class LiveEventToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    tool_name: str
    input: dict[str, JsonValue]


class LiveEventToolCallResponse(BaseModel):
    type: Literal["tool_call_response"] = "tool_call_response"
    tool_call_id: str
    tool_name: str
    output: JsonValue


class LiveEventTurnComplete(BaseModel):
    type: Literal["turn_complete"] = "turn_complete"


type LiveEvent = Annotated[
    LiveEventAudio
    | LiveEventText
    | LiveEventInputTranscription
    | LiveEventOutputTranscription
    | LiveEventInterrupted
    | LiveEventToolCall
    | LiveEventToolCallResponse
    | LiveEventTurnComplete,
    Field(discriminator="type"),
]

__all__ = [
    "LiveEventAudio",
    "LiveEventText",
    "LiveEventInputTranscription",
    "LiveEventOutputTranscription",
    "LiveEventInterrupted",
    "LiveEventToolCall",
    "LiveEventToolCallResponse",
    "LiveEventTurnComplete",
    "LiveEvent",
]
