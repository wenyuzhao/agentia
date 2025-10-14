from datetime import datetime
from typing import Annotated, Literal, Sequence
from pydantic import AliasChoices, BaseModel, Field, JsonValue, HttpUrl

type ProviderOptions = dict[str, dict[str, JsonValue]]
type ProviderMetadata = dict[str, dict[str, JsonValue]]

type DataContent = bytes | str | HttpUrl

type FinishReason = Literal[
    "stop", "length", "content-filter", "tool-calls", "error", "other", "unknown"
]


class ToolChoiceAuto(BaseModel):
    type: Literal["auto"] = "auto"


class ToolChoiceNone(BaseModel):
    type: Literal["none"] = "none"


class ToolChoiceRequired(BaseModel):
    type: Literal["required"] = "required"


class ToolChoiceSpecific(BaseModel):
    type: Literal["tool"] = "tool"
    tool_name: str = Field(
        validation_alias=AliasChoices("toolName", "tool_name"),
        serialization_alias="toolName",
    )


type ToolChoice = Annotated[
    ToolChoiceAuto | ToolChoiceNone | ToolChoiceRequired | ToolChoiceSpecific,
    Field(discriminator="type"),
]


class Usage(BaseModel):
    input_tokens: int | None = Field(
        default=None,
        validation_alias=AliasChoices("inputTokens", "input_tokens"),
        serialization_alias="inputTokens",
    )
    output_tokens: int | None = Field(
        default=None,
        validation_alias=AliasChoices("outputTokens", "output_tokens"),
        serialization_alias="outputTokens",
    )
    total_tokens: int | None = Field(
        default=None,
        validation_alias=AliasChoices("totalTokens", "total_tokens"),
        serialization_alias="totalTokens",
    )
    reasoning_tokens: int | None = Field(
        default=None,
        validation_alias=AliasChoices("reasoningTokens", "reasoning_tokens"),
        serialization_alias="reasoningTokens",
    )
    cached_input_tokens: int | None = Field(
        default=None,
        validation_alias=AliasChoices("cachedInputTokens", "cached_input_tokens"),
        serialization_alias="cachedInputTokens",
    )

    def __add__(self, y: "Usage") -> "Usage":
        x = Usage(**self.model_dump())
        if y.input_tokens:
            x.input_tokens = (x.input_tokens or 0) + y.input_tokens
        if y.output_tokens:
            x.output_tokens = (x.output_tokens or 0) + y.output_tokens
        if y.total_tokens:
            x.total_tokens = (x.total_tokens or 0) + y.total_tokens
        if y.reasoning_tokens:
            x.reasoning_tokens = (x.reasoning_tokens or 0) + y.reasoning_tokens
        if y.cached_input_tokens:
            x.cached_input_tokens = (x.cached_input_tokens or 0) + y.cached_input_tokens
        return x


class ResponseMetadata(BaseModel):
    id: str | None = None
    timestamp: datetime | None = None
    model_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("modelId", "model_id"),
        serialization_alias="modelId",
    )


class FunctionTool(BaseModel):
    type: Literal["function"] = "function"
    name: str
    input_schema: JsonValue = Field(
        validation_alias=AliasChoices("inputSchema", "input_schema"),
        serialization_alias="inputSchema",
    )
    description: str | None = None
    provider_options: ProviderOptions | None = Field(
        default=None,
        validation_alias=AliasChoices("providerOptions", "provider_options"),
        serialization_alias="providerOptions",
    )


class ProviderDefinedTool(BaseModel):
    type: Literal["provider-defined"] = "provider-defined"
    id: str
    name: str
    args: dict[str, JsonValue]


type Tool = Annotated[FunctionTool | ProviderDefinedTool, Field(discriminator="type")]


class UnsupportedSettingWarning(BaseModel):
    type: Literal["unsupported-setting"] = "unsupported-setting"
    setting: str
    details: str | None = None


class UnsupportedToolWarning(BaseModel):
    type: Literal["unsupported-tool"] = "unsupported-tool"
    tool: Tool
    details: str | None = None


class OtherWarning(BaseModel):
    type: Literal["other"] = "other"
    message: str


type Warning = Annotated[
    UnsupportedSettingWarning | UnsupportedToolWarning | OtherWarning,
    Field(discriminator="type"),
]


class ToolCall(BaseModel):
    type: Literal["tool-call"] = "tool-call"

    tool_call_id: str = Field(
        validation_alias=AliasChoices("toolCallId", "tool_call_id"),
        serialization_alias="toolCallId",
    )
    """The identifier of the tool call. It must be unique across all tool calls."""

    tool_name: str = Field(
        validation_alias=AliasChoices("toolName", "tool_name"),
        serialization_alias="toolName",
    )
    """The name of the tool that should be called."""

    input: str
    """Stringified JSON object with the tool call arguments. Must match the parameters schema of the tool."""

    provider_executed: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("providerExecuted", "provider_executed"),
        serialization_alias="providerExecuted",
    )
    """
    Whether the tool call will be executed by the provider.
    If this flag is not set or is false, the tool call will be executed by the client.
    """

    provider_metadata: ProviderMetadata | None = Field(
        default=None,
        validation_alias=AliasChoices("providerMetadata", "provider_metadata"),
        serialization_alias="providerMetadata",
    )
    """Additional provider-specific metadata for the tool call."""


class ToolResult(BaseModel):
    type: Literal["tool-result"] = "tool-result"

    tool_call_id: str = Field(
        validation_alias=AliasChoices("toolCallId", "tool_call_id"),
        serialization_alias="toolCallId",
    )
    """The ID of the tool call that this result is associated with."""

    tool_name: str = Field(
        validation_alias=AliasChoices("toolName", "tool_name"),
        serialization_alias="toolName",
    )
    """Name of the tool that generated this result."""

    result: JsonValue
    """Result of the tool call. This is a JSON-serializable object."""

    is_error: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("isError", "is_error"),
        serialization_alias="isError",
    )
    """Optional flag if the result is an error or an error message."""

    provider_executed: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("providerExecuted", "provider_executed"),
        serialization_alias="providerExecuted",
    )
    """
    Whether the tool result was generated by the provider.
    If this flag is set to true, the tool result was generated by the provider.
    If this flag is not set or is false, the tool result was generated by the client.
    """

    preliminary: bool | None = None
    """
    Whether the tool result is preliminary.

    Preliminary tool results replace each other, e.g. image previews.
    There always has to be a final, non-preliminary tool result.

    If this flag is set to true, the tool result is preliminary.
    If this flag is not set or is false, the tool result is not preliminary.
    """

    provider_metadata: ProviderMetadata | None = Field(
        default=None,
        validation_alias=AliasChoices("providerMetadata", "provider_metadata"),
        serialization_alias="providerMetadata",
    )
    """Additional provider-specific metadata for the tool result."""


class File(BaseModel):
    type: Literal["file"] = "file"

    media_type: str = Field(
        validation_alias=AliasChoices("mediaType", "media_type"),
        serialization_alias="mediaType",
    )
    """
    The IANA media type of the file, e.g. `image/png` or `audio/mp3`.

    @see https://www.iana.org/assignments/media-types/media-types.xhtml
    """

    data: str | bytes
    """
    Generated file data as base64 encoded strings or binary data.

    The file data should be returned without any unnecessary conversion.
    If the API returns base64 encoded strings, the file data should be returned
    as base64 encoded strings. If the API returns binary data, the file data should
    be returned as binary data.
    """


class SourceURL(BaseModel):
    type: Literal["source"] = "source"
    source_type: Literal["url"] = Field(
        "url",
        validation_alias=AliasChoices("sourceType", "source_type"),
        serialization_alias="sourceType",
    )
    id: str
    url: str
    title: str | None = None
    provider_metadata: ProviderMetadata | None = Field(
        default=None,
        validation_alias=AliasChoices("providerMetadata", "provider_metadata"),
        serialization_alias="providerMetadata",
    )


class SourceDocument(BaseModel):
    type: Literal["source"] = "source"
    source_type: Literal["document"] = Field(
        "document",
        validation_alias=AliasChoices("sourceType", "source_type"),
        serialization_alias="sourceType",
    )
    id: str
    media_type: str = Field(
        validation_alias=AliasChoices("mediaType", "media_type"),
        serialization_alias="mediaType",
    )
    title: str
    filename: str | None = None
    provider_metadata: ProviderMetadata | None = Field(
        default=None,
        validation_alias=AliasChoices("providerMetadata", "provider_metadata"),
        serialization_alias="providerMetadata",
    )


type Source = Annotated[SourceURL | SourceDocument, Field(discriminator="source_type")]


class Reasoning(BaseModel):
    type: Literal["reasoning"] = "reasoning"
    text: str
    provider_metadata: ProviderMetadata | None = Field(
        default=None,
        validation_alias=AliasChoices("providerMetadata", "provider_metadata"),
        serialization_alias="providerMetadata",
    )


class Text(BaseModel):
    type: Literal["text"] = "text"
    text: str
    provider_metadata: ProviderMetadata | None = Field(
        default=None,
        validation_alias=AliasChoices("providerMetadata", "provider_metadata"),
        serialization_alias="providerMetadata",
    )


type Content = Annotated[
    Text | Reasoning | File | Source | ToolCall | ToolResult,
    Field(discriminator="type"),
]
