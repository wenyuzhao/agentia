from enum import Enum
import inspect
from typing import Type
from .base import *
from openai.lib._parsing._completions import to_strict_json_schema  # type: ignore


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

    @property
    def text(self) -> str:
        text = ""
        for part in self.content:
            if isinstance(part, MessagePartText):
                text += part.text
        return text


type ObjectType = BaseModel | Enum | str | int | float | bool | None | list[
    "ObjectType"
] | dict[str, "ObjectType"] | tuple["ObjectType", ...]


class _Result[X](BaseModel):
    result: X = Field(..., description="The result", title="Result")


class AssistantMessage(MessageBase):
    role: Literal["assistant"] = "assistant"
    content: Sequence[AssistantMessagePart]

    @property
    def reasoning(self) -> str | None:
        text = ""
        for part in self.content:
            if isinstance(part, MessagePartReasoning):
                text += part.text
        return text

    @property
    def text(self) -> str:
        text = ""
        for part in self.content:
            if isinstance(part, MessagePartText):
                text += part.text
        return text

    def parse[T: ObjectType](self, return_type: type[T]) -> T:
        class Result(_Result[return_type]): ...

        if inspect.isclass(return_type) and issubclass(return_type, BaseModel):
            response_format = return_type
        else:
            response_format = Result

        json_string = ""
        for part in self.content:
            if isinstance(part, MessagePartText):
                json_string += part.text

        result = response_format.model_validate_json(json_string)
        if isinstance(result, _Result):
            return result.result
        else:
            return result

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

type NonSystemMessage = Annotated[
    UserMessage | AssistantMessage | ToolMessage,
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

    @staticmethod
    def from_model[T: ObjectType](return_type: Type[T]) -> "ResponseFormatJson":
        class Result(_Result[return_type]): ...

        if inspect.isclass(return_type) and issubclass(return_type, BaseModel):
            response_format = return_type
        else:
            response_format = Result
        return ResponseFormatJson(
            json_schema=to_strict_json_schema(response_format),
            name=return_type.__name__,
            description=f"JSON object matching the schema of {return_type.__name__}",
        )


type ResponseFormat = Annotated[
    ResponseFormatText | ResponseFormatJson,
    Field(discriminator="type"),
]
