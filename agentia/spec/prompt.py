from attr import dataclass
from .base import *


class MessageBase(BaseModel):
    provider_options: ProviderOptions | None = Field(
        default=None,
        validation_alias=AliasChoices("providerOptions", "providerOptions"),
        serialization_alias="providerOptions",
    )


class SystemMessage(MessageBase):
    role: Literal["system"] = "system"
    content: str


class MessagePartBase(BaseModel):
    provider_options: ProviderOptions | None = Field(
        default=None,
        validation_alias=AliasChoices("providerOptions", "provider_options"),
        serialization_alias="providerOptions",
    )


class MessagePartText(MessagePartBase):
    type: Literal["text"] = "text"
    text: str


class MessagePartReasoning(MessagePartBase):
    type: Literal["reasoning"] = "reasoning"
    text: str


class MessagePartFile(MessagePartBase):
    type: Literal["file"] = "file"
    data: DataContent
    media_type: str = Field(
        validation_alias=AliasChoices("mediaType", "media_type"),
        serialization_alias="mediaType",
    )
    filename: str | None = None


class MessagePartToolCall(BaseModel):
    type: Literal["tool-call"] = "tool-call"
    tool_call_id: str = Field(
        validation_alias=AliasChoices("toolCallId", "tool_call_id"),
        serialization_alias="toolCallId",
    )
    tool_name: str = Field(
        validation_alias=AliasChoices("toolName", "tool_name"),
        serialization_alias="toolName",
    )
    input: JsonValue
    provider_executed: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("providerExecuted", "provider_executed"),
        serialization_alias="providerExecuted",
    )
    provider_options: ProviderOptions | None = Field(
        default=None,
        validation_alias=AliasChoices("providerOptions", "provider_options"),
        serialization_alias="providerOptions",
    )


class ToolResultOutputText(BaseModel):
    type: Literal["text"] = "text"
    value: str


class ToolResultOutputJson(BaseModel):
    type: Literal["json"] = "json"
    value: JsonValue


class ToolResultOutputExecutionDenied(BaseModel):
    type: Literal["execution-denied"] = "execution-denied"
    reason: str | None = None


class ToolResultOutputErrorText(BaseModel):
    type: Literal["error-text"] = "error-text"
    value: str


class ToolResultOutputErrorJson(BaseModel):
    type: Literal["error-json"] = "error-json"
    value: JsonValue


class ToolResultOutputContentPartText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolResultOutputContentPartMedia(BaseModel):
    type: Literal["media"] = "media"
    data: str
    media_type: str = Field(
        validation_alias=AliasChoices("mediaType", "media_type"),
        serialization_alias="mediaType",
    )


type ToolResultOutputContentPart = Annotated[
    ToolResultOutputContentPartText | ToolResultOutputContentPartMedia,
    Field(discriminator="type"),
]


class ToolResultOutputContent(BaseModel):
    type: Literal["content"] = "content"
    value: Sequence[ToolResultOutputContentPart]


type ToolResultOutput = Annotated[
    ToolResultOutputText
    | ToolResultOutputJson
    | ToolResultOutputExecutionDenied
    | ToolResultOutputErrorText
    | ToolResultOutputErrorJson
    | ToolResultOutputContent,
    Field(discriminator="type"),
]


class MessagePartToolResult(BaseModel):
    type: Literal["tool-result"] = "tool-result"
    tool_call_id: str = Field(
        validation_alias=AliasChoices("toolCallId", "tool_call_id"),
        serialization_alias="toolCallId",
    )
    tool_name: str = Field(
        validation_alias=AliasChoices("toolName", "tool_name"),
        serialization_alias="toolName",
    )
    output: ToolResultOutput
    provider_options: ProviderOptions | None = Field(
        default=None,
        validation_alias=AliasChoices("providerOptions", "provider_options"),
        serialization_alias="providerOptions",
    )


type MessagePart = Annotated[
    MessagePartText
    | MessagePartReasoning
    | MessagePartFile
    | MessagePartToolCall
    | MessagePartToolResult,
    Field(discriminator="type"),
]

type UserMessagePart = Annotated[
    MessagePartText | MessagePartFile,
    Field(discriminator="type"),
]

type AssistantMessagePart = Annotated[
    MessagePartText
    | MessagePartReasoning
    | MessagePartFile
    | MessagePartToolCall
    | MessagePartToolResult,
    Field(discriminator="type"),
]


class UserMessage(MessageBase):
    role: Literal["user"] = "user"
    content: Sequence[UserMessagePart]


class AssistantMessage(MessageBase):
    role: Literal["assistant"] = "assistant"
    content: Sequence[AssistantMessagePart]

    def get_text_content(self) -> str:
        text = ""
        for part in self.content:
            if isinstance(part, MessagePartText):
                text += part.text
        return text

    @staticmethod
    def from_contents(contents: Sequence[Content]) -> "AssistantMessage":
        parts = []
        for c in contents:
            if c.type == "text":
                parts.append(MessagePartText(text=c.text))
            elif c.type == "reasoning":
                parts.append(MessagePartReasoning(text=c.text))
            elif c.type == "file":
                parts.append(MessagePartFile(data=c.data, media_type=c.media_type))
            elif c.type == "tool-call":
                parts.append(
                    MessagePartToolCall(
                        tool_call_id=c.tool_call_id,
                        tool_name=c.tool_name,
                        input=c.input,
                        provider_executed=c.provider_executed,
                    )
                )
            else:
                raise Exception(f"Unknown content type: {c.type}")
        assert len(parts) > 0
        return AssistantMessage(content=parts)


class ToolMessage(MessageBase):
    role: Literal["tool"] = "tool"
    content: Sequence[MessagePartToolResult]


type Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role"),
]

type Prompt = Sequence[Message]


class ResponseFormatText(BaseModel):
    type: Literal["text"] = "text"


class ResponseFormatJson(BaseModel):
    type: Literal["json"] = "json"
    json_schema: JsonValue | None = Field(
        default=None,
        validation_alias=AliasChoices("schema", "json_schema"),
        serialization_alias="schema",
    )
    name: str | None = None
    description: str | None = None


type ResponseFormat = Annotated[
    ResponseFormatText | ResponseFormatJson,
    Field(discriminator="type"),
]
