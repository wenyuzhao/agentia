from io import BytesIO, StringIO
from pathlib import Path
from typing import (
    Literal,
    Required,
    TypeAlias,
    TypeVar,
    TypedDict,
    Union,
    cast,
    Mapping,
    Sequence,
    Any,
    override,
    TYPE_CHECKING,
)
import json
import abc
from dataclasses import dataclass, field
import uuid

from pydantic import BaseModel


if TYPE_CHECKING:
    from agentia.tools import ClientTool
    from openai.types.chat import (
        ChatCompletionContentPartTextParam,
        ChatCompletionContentPartImageParam,
    )


JSON: TypeAlias = (
    Mapping[str, "JSON"] | Sequence["JSON"] | str | int | float | bool | None
)


class FunctionCallDict(TypedDict, total=False):
    arguments: Required[str]
    """
    The arguments to call the function with, as generated by the model in JSON
    format. Note that the model does not always generate valid JSON, and may
    hallucinate parameters not defined by your function schema. Validate the
    arguments in your code before calling your function.
    """

    name: Required[str]
    """The name of the function to call."""


@dataclass
class FunctionCall:
    name: str
    arguments: JSON

    def arguments_string(self) -> str:
        return json.dumps(self.arguments)

    def to_dict(self) -> FunctionCallDict:
        return {
            "name": self.name,
            "arguments": self.arguments_string(),
        }

    @staticmethod
    def from_dict(d: FunctionCallDict) -> "FunctionCall":
        return FunctionCall(
            name=d["name"],
            arguments=json.loads(d["arguments"]),
        )


@dataclass
class ToolCall:
    id: str
    function: FunctionCall
    type: Literal["function"]

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ToolCall":
        return ToolCall(
            id=d["id"],
            function=FunctionCall.from_dict(d["function"]),
            type=d["type"],
        )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "id": self.id,
            "function": self.function.to_dict(),
            "type": self.type,
        }


Role: TypeAlias = Literal["system", "user", "assistant", "tool"]


class ContentPartText:
    def __init__(self, content: str) -> None:
        self.content = content

    def to_openai_content_part(self) -> "ChatCompletionContentPartTextParam":
        return {"type": "text", "text": self.content}


class ContentPartImage:
    def __init__(self, url: str) -> None:
        self.url = url

    def to_openai_content_part(self) -> "ChatCompletionContentPartImageParam":
        return {"type": "image_url", "image_url": {"url": self.url}}


ContentPart = Union[ContentPartText, ContentPartImage]


@dataclass
class BaseMessage(abc.ABC):
    @abc.abstractmethod
    def to_json(self) -> Mapping[str, Any]:
        raise NotImplementedError()

    @staticmethod
    def from_json(data: Any) -> "Message":
        data = cast(Mapping[str, Any], data)
        assert data["role"] in ["system", "user", "assistant", "tool"]
        match data["role"]:
            case "user":
                return UserMessage(
                    content=(
                        data["content"]
                        if isinstance(data["content"], str) or data["content"] is None
                        else [
                            (
                                ContentPartText(c["text"])
                                if c["type"] == "text"
                                else ContentPartImage(c["image_url"]["url"])
                            )
                            for c in data["content"]
                        ]
                    ),
                    name=data.get("name"),
                )
            case "system":
                return SystemMessage(content=data["content"])
            case "assistant":
                return AssistantMessage(
                    content=data["content"] or "",
                    tool_calls=[
                        ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])
                    ],
                )
            case "tool":
                return ToolMessage(
                    content=data["content"],
                    tool_call_id=data.get("tool_call_id"),
                )
            case _:
                raise ValueError("Invalid role: " + data["role"])


@dataclass
class UserMessage(BaseMessage):
    content: str | Sequence[ContentPart]
    """
    The contents of the message.

    `content` is required for all messages, and may be null for assistant messages
    with function calls.
    """

    name: str | None = None
    """
    Used for function messages to indicate the name of the function that was called.
    Function return data is provided in the `content` field.
    """

    files: Sequence[str | Path | BytesIO | StringIO] = field(default_factory=list)

    role: Literal["user"] = "user"

    @override
    def to_json(self) -> Mapping[str, Any]:
        data: Mapping[str, Any] = {
            "role": "user",
            "content": (
                self.content
                if isinstance(self.content, str) or self.content is None
                else [c.to_openai_content_part() for c in self.content]
            ),
        }
        if self.name is not None:
            data["name"] = self.name
        return data


@dataclass
class SystemMessage(BaseMessage):
    content: str
    """
    The contents of the message.

    `content` is required for all messages, and may be null for assistant messages
    with function calls.
    """

    role: Literal["system"] = "system"

    @override
    def to_json(self) -> Mapping[str, Any]:
        data: Mapping[str, Any] = {"role": "system", "content": self.content}
        return data


T = TypeVar("T", bound=BaseModel)


@dataclass
class AssistantMessage(BaseMessage):
    content: str
    """
    The contents of the message.

    `content` is required for all messages, and may be null for assistant messages
    with function calls.
    """

    reasoning: str | None = None
    """The reasoning behind the assistant's response."""

    tool_calls: Sequence[ToolCall] = field(default_factory=list)
    """The tool calls generated by the model, such as function calls."""

    role: Literal["assistant"] = "assistant"

    @override
    def to_json(self) -> Mapping[str, Any]:
        data: Mapping[str, Any] = {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }
        return data

    def cast(self, t: type[T]) -> T:
        if t == str:
            try:
                parsed = json.loads(self.content)
                if isinstance(parsed, str):
                    return parsed  # type: ignore
            except json.JSONDecodeError:
                return str(self.content)  # type: ignore
        parsed = json.loads(self.content)
        if issubclass(t, BaseModel):
            return t(**parsed)
        else:

            class Value[V](BaseModel):
                value: V

            return Value[T](value=parsed).value


@dataclass
class ToolMessage:
    content: str
    """
    The contents of the message.

    `content` is required for all messages, and may be null for assistant messages
    with function calls.
    """

    tool_call_id: str | None = None
    """Tool call that this message is responding to."""

    role: Literal["tool"] = "tool"

    def to_json(self) -> Mapping[str, Any]:
        data: Mapping[str, Any] = {
            "role": "tool",
            "content": self.content,
            "tool_call_id": self.tool_call_id,
        }
        return data


Message: TypeAlias = UserMessage | SystemMessage | AssistantMessage | ToolMessage


@dataclass
class ToolCallEvent:
    id: str
    agent: str
    name: str
    display_name: str
    description: str
    arguments: dict[str, Any]
    result: Any | None = None
    metadata: Any | None = None
    role: Literal["event.tool_call"] = "event.tool_call"


@dataclass
class CommunicationEvent:
    id: str
    parent: str
    child: str
    message: str
    response: str | None = None
    role: Literal["event.communication"] = "event.communication"


@dataclass
class UserConsentEvent:
    message: str
    response: bool | None = None
    metadata: Any | None = None
    tool: str | None = None
    plugin: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: Literal["event.user_consent"] = "event.user_consent"


@dataclass
class ClientToolCallEvent:
    tool: "ClientTool"
    args: dict[str, Any]
    response: Any | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: Literal["event.user_consent"] = "event.user_consent"


Event: TypeAlias = (
    ToolCallEvent | CommunicationEvent | UserConsentEvent | ClientToolCallEvent
)

__all__ = [
    "ToolCall",
    "FunctionCall",
    "FunctionCallDict",
    "ContentPart",
    "ContentPartText",
    "ContentPartImage",
    "Role",
    # Message types
    "Message",
    "UserMessage",
    "SystemMessage",
    "AssistantMessage",
    "ToolMessage",
    # Event types
    "Event",
    "CommunicationEvent",
    "UserConsentEvent",
    "ToolCallEvent",
    "ClientToolCallEvent",
]
