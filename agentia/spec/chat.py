from enum import Enum
import inspect
from typing import Annotated, Any, Literal, Sequence, Type
from .base import *
from openai.lib._parsing._completions import to_strict_json_schema  # type: ignore
from pydantic import AliasChoices, BaseModel, Field, JsonValue
import json


class MessageBase(BaseModel):
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)


class SystemMessage(MessageBase):
    role: Literal["system"] = "system"
    content: str
    provider_options: ProviderOptions | None = Field(
        default=None,
        validation_alias=AliasChoices("providerOptions", "providerOptions"),
        serialization_alias="providerOptions",
    )

    def __init__(
        self,
        content: str,
        *,
        role: Literal["system"] = "system",
        provider_options: ProviderOptions | None = None,
    ):
        super().__init__(role=role, content=content, provider_options=provider_options)


class MessagePartBase(BaseModel):
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)


class MessagePartText(MessagePartBase):
    type: Literal["text"] = "text"
    text: str
    provider_options: ProviderOptions | None = Field(
        default=None,
        validation_alias=AliasChoices("providerOptions", "provider_options"),
        serialization_alias="providerOptions",
    )

    def __init__(
        self,
        text: str,
        *,
        type: Literal["text"] = "text",
        provider_options: ProviderOptions | None = None,
    ):
        super().__init__(type=type, text=text, provider_options=provider_options)


class MessagePartReasoning(MessagePartBase):
    type: Literal["reasoning"] = "reasoning"
    text: str
    provider_options: ProviderOptions | None = Field(
        default=None,
        validation_alias=AliasChoices("providerOptions", "provider_options"),
        serialization_alias="providerOptions",
    )

    def __init__(
        self,
        text: str,
        *,
        type: Literal["reasoning"] = "reasoning",
        provider_options: ProviderOptions | None = None,
    ):
        super().__init__(type=type, text=text, provider_options=provider_options)


class MessagePartFile(MessagePartBase):
    type: Literal["file"] = "file"
    data: DataContent
    media_type: str = Field(
        validation_alias=AliasChoices("mediaType", "media_type"),
        serialization_alias="mediaType",
    )
    filename: str | None = None
    provider_options: ProviderOptions | None = Field(
        default=None,
        validation_alias=AliasChoices("providerOptions", "provider_options"),
        serialization_alias="providerOptions",
    )

    def to_url(self) -> str:
        return File(data=self.data, media_type=self.media_type).to_url()


class MessagePartToolCall(MessagePartBase):
    type: Literal["tool-call"] = "tool-call"
    tool_call_id: str = Field(
        validation_alias=AliasChoices("toolCallId", "tool_call_id"),
        serialization_alias="toolCallId",
    )
    tool_name: str = Field(
        validation_alias=AliasChoices("toolName", "tool_name"),
        serialization_alias="toolName",
    )
    input: dict[str, JsonValue]
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


class MessagePartToolResult(MessagePartBase):
    type: Literal["tool-result"] = "tool-result"
    tool_call_id: str = Field(
        validation_alias=AliasChoices("toolCallId", "tool_call_id"),
        serialization_alias="toolCallId",
    )
    tool_name: str = Field(
        validation_alias=AliasChoices("toolName", "tool_name"),
        serialization_alias="toolName",
    )
    input: dict[str, JsonValue]
    output: JsonValue
    output_files: list[File] | None = Field(
        default=None,
        validation_alias=AliasChoices("outputFiles", "output_files"),
        serialization_alias="outputFiles",
    )
    provider_options: ProviderOptions | None = Field(
        default=None,
        validation_alias=AliasChoices("providerOptions", "provider_options"),
        serialization_alias="providerOptions",
    )

    def serialize_output(self) -> str:
        return (
            json.dumps(self.output)
            if isinstance(self.output, (list, dict))
            else str(self.output)
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
    MessagePartText | MessagePartFile, Field(discriminator="type")
]

type AssistantMessagePart = Annotated[
    MessagePartText | MessagePartReasoning | MessagePartToolCall,
    Field(discriminator="type"),
]


class UserMessage(MessageBase):
    role: Literal["user"] = "user"
    content: Sequence[UserMessagePart] | UserMessagePart | str
    provider_options: ProviderOptions | None = Field(
        default=None,
        validation_alias=AliasChoices("providerOptions", "providerOptions"),
        serialization_alias="providerOptions",
    )

    def __init__(
        self,
        content: str | Sequence[UserMessagePart] | UserMessagePart,
        *,
        role: Literal["user"] = "user",
        provider_options: ProviderOptions | None = None,
    ):
        super().__init__(role=role, content=content, provider_options=provider_options)

    @property
    def content_list(self) -> list[UserMessagePart]:
        if isinstance(self.content, str):
            return [MessagePartText(text=self.content)]
        if not isinstance(self.content, (Sequence, list)):
            return [self.content]
        return list(self.content)

    @property
    def text(self) -> str:
        text = ""
        for part in self.content_list:
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
    content: Sequence[AssistantMessagePart] | AssistantMessagePart | str
    annotations: list[Annotation] | None = None
    provider_options: ProviderOptions | None = Field(
        default=None,
        validation_alias=AliasChoices("providerOptions", "providerOptions"),
        serialization_alias="providerOptions",
    )

    def __init__(
        self,
        content: str | Sequence[AssistantMessagePart] | AssistantMessagePart,
        *,
        role: Literal["assistant"] = "assistant",
        annotations: list[Annotation] | None = None,
        provider_options: ProviderOptions | None = None,
    ):
        super().__init__(role=role, content=content, provider_options=provider_options)

    @property
    def content_list(self) -> list[AssistantMessagePart]:
        if isinstance(self.content, str):
            return [MessagePartText(text=self.content)]
        if not isinstance(self.content, Sequence):
            return [self.content]
        return list(self.content)

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
        for part in self.content_list:
            if isinstance(part, MessagePartText):
                text += part.text
        return text

    @property
    def tool_calls(self) -> list[ToolCall]:
        tool_calls = []
        for part in self.content_list:
            if isinstance(part, MessagePartToolCall):
                tool_calls.append(
                    ToolCall(
                        tool_call_id=part.tool_call_id,
                        tool_name=part.tool_name,
                        input=part.input,
                        provider_executed=part.provider_executed,
                    )
                )
        return tool_calls

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


class ToolMessage(MessageBase):
    role: Literal["tool"] = "tool"
    content: Sequence[MessagePartToolResult]
    provider_options: ProviderOptions | None = Field(
        default=None,
        validation_alias=AliasChoices("providerOptions", "providerOptions"),
        serialization_alias="providerOptions",
    )

    def __init__(
        self,
        content: Sequence[MessagePartToolResult],
        *,
        role: Literal["tool"] = "tool",
        provider_options: ProviderOptions | None = None,
    ):
        super().__init__(role=role, content=content, provider_options=provider_options)


type Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role"),
]

type NonSystemMessage = Annotated[
    UserMessage | AssistantMessage | ToolMessage, Field(discriminator="role")
]


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
    ResponseFormatText | ResponseFormatJson, Field(discriminator="type")
]


__all__ = [
    "MessageBase",
    "SystemMessage",
    "MessagePartBase",
    "MessagePartText",
    "MessagePartReasoning",
    "MessagePartFile",
    "MessagePartToolCall",
    "MessagePartToolResult",
    "MessagePart",
    "UserMessagePart",
    "AssistantMessagePart",
    "UserMessage",
    "ObjectType",
    "AssistantMessage",
    "ToolMessage",
    "Message",
    "NonSystemMessage",
    "ResponseFormatText",
    "ResponseFormatJson",
    "ResponseFormat",
]
