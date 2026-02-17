from enum import Enum
import inspect
from typing import Annotated, Any, Literal, Sequence, Type
from .base import *
from openai.lib._parsing._completions import to_strict_json_schema  # type: ignore
from pydantic import BaseModel, Field, HttpUrl, JsonValue
import json


class MessageBase(BaseModel):
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)


class SystemMessage(MessageBase):
    role: Literal["system"] = "system"
    content: str
    provider_options: ProviderOptions | None = None

    def __init__(
        self,
        content: str,
        *,
        role: Literal["system"] = "system",
        provider_options: ProviderOptions | None = None,
    ):
        super().__init__(role=role, content=content, provider_options=provider_options)

    def to_openai_format(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


class MessagePartBase(BaseModel):
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)


class MessagePartText(MessagePartBase):
    type: Literal["text"] = "text"
    text: str
    provider_options: ProviderOptions | None = None

    def __init__(
        self,
        text: str,
        *,
        type: Literal["text"] = "text",
        provider_options: ProviderOptions | None = None,
    ):
        super().__init__(type=type, text=text, provider_options=provider_options)

    def to_openai_format(self) -> dict[str, Any]:
        return {"type": self.type, "text": self.text}


class MessagePartReasoning(MessagePartBase):
    type: Literal["reasoning"] = "reasoning"
    text: str
    provider_options: ProviderOptions | None = None

    def __init__(
        self,
        text: str,
        *,
        type: Literal["reasoning"] = "reasoning",
        provider_options: ProviderOptions | None = None,
    ):
        super().__init__(type=type, text=text, provider_options=provider_options)

    def to_openai_format(self) -> dict[str, Any]:
        return {"type": self.type, "text": self.text}


class MessagePartFile(MessagePartBase):
    type: Literal["file"] = "file"
    data: DataContent
    media_type: str
    filename: str | None = None
    provider_options: ProviderOptions | None = None

    def to_url(self) -> str:
        return File(data=self.data, media_type=self.media_type).to_url()

    def to_openai_format(self) -> dict[str, Any]:
        if self.media_type.startswith("image/"):
            url = self.to_url()
            detail = (self.provider_options or {}).get("imageDetail", None)
            if detail not in ["auto", "low", "high"]:
                detail = "auto"
            return {"type": "image_url", "image_url": {"url": url, "detail": detail}}
        if self.media_type.startswith("video/"):
            url = self.to_url()
            return {"type": "video_url", "video_url": {"url": url}}  # type: ignore
        elif self.media_type.startswith("audio/"):
            if isinstance(self.data, HttpUrl) or (
                isinstance(self.data, str)
                and (
                    self.data.startswith("http://") or self.data.startswith("https://")
                )
            ):
                raise ValueError("audio file parts with URLs")
            if isinstance(self.data, str):
                data = self.data
            else:
                assert isinstance(self.data, bytes)
                data = self.data.decode(encoding="utf-8")
            format: Literal["wav", "mp3"]
            if self.media_type == "audio/wav":
                format = "wav"
            elif self.media_type == "audio/mpeg" or self.media_type == "audio/mp3":
                format = "mp3"
            else:
                raise ValueError(f"audio file parts with media type {self.media_type}")
            return {
                "type": "input_audio",
                "input_audio": {"data": data, "format": format},
            }
        else:
            if isinstance(self.data, str) and self.data.startswith("file-"):
                # this is a file ID
                return {"type": "file", "file": {"file_id": self.data}}
            else:
                url = self.to_url()
                return {
                    "type": "file",
                    "file": {"filename": self.filename, "file_data": url},
                }


class MessagePartToolCall(MessagePartBase):
    type: Literal["tool-call"] = "tool-call"
    tool_call_id: str
    tool_name: str
    input: dict[str, JsonValue]
    provider_executed: bool | None = None
    provider_options: ProviderOptions | None = None


class MessagePartToolResult(MessagePartBase):
    type: Literal["tool-result"] = "tool-result"
    tool_call_id: str
    tool_name: str
    input: dict[str, JsonValue]
    output: JsonValue
    output_files: list[File] | None = None
    provider_options: ProviderOptions | None = None

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
    content: Sequence[UserMessagePart] | str
    provider_options: ProviderOptions | None = None

    def __init__(
        self,
        content: str | Sequence[UserMessagePart],
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

    def to_openai_format(self) -> dict[str, Any]:
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        else:
            return {
                "role": self.role,
                "content": [part.to_openai_format() for part in self.content_list],
            }


type ObjectType = BaseModel | Enum | str | int | float | bool | None | list[
    "ObjectType"
] | dict[str, "ObjectType"] | tuple["ObjectType", ...]


class _Result[X](BaseModel):
    result: X = Field(..., description="The result", title="Result")


class AssistantMessage(MessageBase):
    role: Literal["assistant"] = "assistant"
    content: Sequence[AssistantMessagePart] | str
    annotations: list[Annotation] | None = None
    provider_options: ProviderOptions | None = None

    def __init__(
        self,
        content: str | Sequence[AssistantMessagePart],
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
        for part in self.content_list:
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

    def to_openai_format(self) -> dict[str, Any]:
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        else:
            text_parts: list[str] = []
            images: list[Any] = []
            tool_calls: list[Any] = []
            reasoning = ""
            for i, p in enumerate(self.content_list):
                if p.type == "text":
                    text_parts.append(p.text)
                elif p.type == "reasoning":
                    reasoning += p.text
                elif p.type == "file":
                    if p.media_type.startswith("image/"):
                        images.append(
                            {"type": "image_url", "image_url": {"url": p.to_url()}}
                        )
                    else:
                        raise ValueError(
                            f"Only image files are supported in assistant messages, got media type {p.media_type}"
                        )
                elif p.type == "tool-call":
                    tool_calls.append(
                        {
                            "id": p.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": p.tool_name,
                                "arguments": json.dumps(p.input),
                            },
                        }
                    )
            p = {
                "role": "assistant",
                "content": (
                    text_parts[0]
                    if len(text_parts) == 1
                    else text_parts if text_parts else None
                ),
                "tool_calls": tool_calls,
            }
            if reasoning:
                p["reasoning"] = reasoning
            if images:
                p["images"] = images
            return p


class ToolMessage(MessageBase):
    role: Literal["tool"] = "tool"
    content: Sequence[MessagePartToolResult]
    provider_options: ProviderOptions | None = None

    def __init__(
        self,
        content: Sequence[MessagePartToolResult],
        *,
        role: Literal["tool"] = "tool",
        provider_options: ProviderOptions | None = None,
    ):
        super().__init__(role=role, content=content, provider_options=provider_options)

    def to_openai_format(self) -> dict[str, Any] | list[dict[str, Any]]:
        files: list[File] = []
        msgs = []
        for p in self.content:
            if p.output_files:
                p.output = {
                    "output": p.output,
                    "files_hint": f"The tool output includes {len(p.output_files)} file(s).",
                    "files": [f.model_dump() for f in p.output_files],
                }
                files.extend(p.output_files)
            val = p.serialize_output()
            msgs.append(
                {"role": "tool", "tool_call_id": p.tool_call_id, "content": val}
            )
        if files:
            user_msg = UserMessage(
                content=[
                    MessagePartText(text="[[TOOL_OUTPUT_FILES]]"),
                    *[
                        MessagePartFile(media_type=f.media_type, data=f.to_url())
                        for f in files
                    ],
                ]
            )
            msgs.append(user_msg.to_openai_format())
        return msgs if len(msgs) > 1 else msgs[0]


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
    json_schema: JsonValue | None = None
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
