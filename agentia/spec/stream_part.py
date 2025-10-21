from datetime import datetime
from typing import Annotated, Literal, Sequence
from pydantic import AliasChoices, BaseModel, Field, JsonValue
from .base import *


class StreamPartTextStart(BaseModel):
    type: Literal["text-start"] = "text-start"
    id: str
    provider_metadata: ProviderMetadata | None = Field(
        default=None,
        validation_alias=AliasChoices("providerMetadata", "provider_metadata"),
        serialization_alias="providerMetadata",
    )


class StreamPartTextDelta(BaseModel):
    type: Literal["text-delta"] = "text-delta"
    delta: str
    id: str
    provider_metadata: ProviderMetadata | None = Field(
        default=None,
        validation_alias=AliasChoices("providerMetadata", "provider_metadata"),
        serialization_alias="providerMetadata",
    )


class StreamPartTextEnd(BaseModel):
    type: Literal["text-end"] = "text-end"
    id: str
    provider_metadata: ProviderMetadata | None = Field(
        default=None,
        validation_alias=AliasChoices("providerMetadata", "provider_metadata"),
        serialization_alias="providerMetadata",
    )


class StreamPartReasoningStart(BaseModel):
    type: Literal["reasoning-start"] = "reasoning-start"
    id: str
    provider_metadata: ProviderMetadata | None = Field(
        default=None,
        validation_alias=AliasChoices("providerMetadata", "provider_metadata"),
        serialization_alias="providerMetadata",
    )


class StreamPartReasoningDelta(BaseModel):
    type: Literal["reasoning-delta"] = "reasoning-delta"
    delta: str
    id: str
    provider_metadata: ProviderMetadata | None = Field(
        default=None,
        validation_alias=AliasChoices("providerMetadata", "provider_metadata"),
        serialization_alias="providerMetadata",
    )


class StreamPartReasoningEnd(BaseModel):
    type: Literal["reasoning-end"] = "reasoning-end"
    id: str
    provider_metadata: ProviderMetadata | None = Field(
        default=None,
        validation_alias=AliasChoices("providerMetadata", "provider_metadata"),
        serialization_alias="providerMetadata",
    )


class StreamPartToolInputStart(BaseModel):
    type: Literal["tool-input-start"] = "tool-input-start"
    tool_name: str = Field(
        validation_alias=AliasChoices("toolName", "tool_name"),
        serialization_alias="toolName",
    )
    provider_executed: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("providerExecuted", "provider_executed"),
        serialization_alias="providerExecuted",
    )
    id: str
    provider_metadata: ProviderMetadata | None = Field(
        default=None,
        validation_alias=AliasChoices("providerMetadata", "provider_metadata"),
        serialization_alias="providerMetadata",
    )


class StreamPartToolInputDelta(BaseModel):
    type: Literal["tool-input-delta"] = "tool-input-delta"
    delta: str
    id: str
    provider_metadata: ProviderMetadata | None = Field(
        default=None,
        validation_alias=AliasChoices("providerMetadata", "provider_metadata"),
        serialization_alias="providerMetadata",
    )


class StreamPartToolInputEnd(BaseModel):
    type: Literal["tool-input-end"] = "tool-input-end"
    id: str
    provider_metadata: ProviderMetadata | None = Field(
        default=None,
        validation_alias=AliasChoices("providerMetadata", "provider_metadata"),
        serialization_alias="providerMetadata",
    )


class StreamPartStreamStart(BaseModel):
    type: Literal["stream-start"] = "stream-start"
    warnings: Sequence[Warning]


class StreamPartResponseMetadata(ResponseMetadata):
    type: Literal["response-metadata"] = "response-metadata"


class StreamPartFinish(BaseModel):
    type: Literal["finish"] = "finish"
    usage: Usage
    finish_reason: FinishReason = Field(
        validation_alias=AliasChoices("finishReason", "finish_reason"),
        serialization_alias="finishReason",
    )


class StreamPartRaw(BaseModel):
    type: Literal["raw"] = "raw"
    raw_value: JsonValue = Field(
        validation_alias=AliasChoices("rawValue", "raw_value"),
        serialization_alias="rawValue",
    )


class StreamPartError(BaseModel):
    type: Literal["error"] = "error"
    error: JsonValue


type StreamPart = Annotated[
    StreamPartTextStart
    | StreamPartTextDelta
    | StreamPartTextEnd
    | StreamPartReasoningStart
    | StreamPartReasoningDelta
    | StreamPartReasoningEnd
    | StreamPartToolInputStart
    | StreamPartToolInputDelta
    | StreamPartToolInputEnd
    | ToolCall
    | ToolResult
    | File
    | Source
    | StreamPartStreamStart
    | StreamPartResponseMetadata
    | StreamPartFinish
    | StreamPartRaw
    | StreamPartError,
    Field(discriminator="type"),
]
