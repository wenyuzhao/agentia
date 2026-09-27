"""Conversation items -- the unit the history is made of.

Mirrors the Responses API `ItemField` union: a turn is not one message but a sequence of
items (a `reasoning` item, then a `message` and/or one `function_call` per tool), and a
tool result is its own `function_call_output` item.

Responses API types
-------------------
- `Message`            -- `message`. `role` is user / assistant / system / developer,
                          `content` a list of content parts, plus `id` and `status`.
- `FunctionCall`       -- `function_call`, with `call_id`, `name` and `arguments` (a JSON
                          *string*, as the API defines it).
- `FunctionCallOutput` -- `function_call_output`, with `call_id` and `output`. This is an
                          *input* item for the next request, never model output.
- `Reasoning`          -- `reasoning`, with `content` (`reasoning_text`), `summary`
                          (`summary_text`) and `encrypted_content`.
- `Compaction`         -- `compaction`. See the note under extensions.
- `ItemStatus`         -- the `status` field: in_progress / completed / incomplete.
- `MessageRole`        -- the `role` field of a `message`.
- `Item`               -- the `ItemField` union itself, discriminated on `type`.

Agentia extensions
------------------
- `Message.user()` / `.assistant()` / `.system()` / `.developer()` -- constructors that
  coerce plain strings to `input_text` / `output_text` and a `File` to the right content
  part. Plus the `text`, `refusal` accessors and `parse()` for structured output.
- `FunctionCall.create()` / `.input` -- build from, and parse back to, a dict instead of
  the API's `arguments` JSON string.
- `FunctionCallOutput.create()` / `.text` -- JSON-encode structured output and attach
  files. Its `name` and `is_error` fields are agentia-only and are NOT sent to the
  provider; the API's output item carries neither.
- `Reasoning.signature` and `.format` -- provider extensions (OpenRouter passes these
  through for Anthropic models), not part of the OpenAI schema. `.text`,
  `.summary_text` and `.display_text` are convenience accessors.
- `Compaction` semantics -- the API defines `compaction` as an opaque, server-held item
  produced by its compaction endpoint. Agentia compacts client-side instead, so
  `summary` is plain text and `original_items` keeps what was replaced; it serializes as
  an assistant `message`, not as a `compaction` item.
- `to_responses_input()` -- flatten items into the `input` array, dropping items that
  cannot be replayed (a `Reasoning` with no `id`).
- `ObjectType`, `MessageInput`, `InputItem` -- type aliases for the helper signatures.
"""

import inspect
import json
from enum import Enum
from typing import Annotated, Any, Literal, Self, Sequence
from uuid import uuid4

from pydantic import BaseModel, Field, JsonValue

from .base import File
from .content import (
    Content,
    InputContent,
    InputText,
    OutputText,
    ReasoningText,
    Refusal,
    SummaryText,
    input_content_from_file,
)

type ItemStatus = Literal["in_progress", "completed", "incomplete"]

type MessageRole = Literal["user", "assistant", "system", "developer"]

type ObjectType = (
    BaseModel
    | Enum
    | str
    | int
    | float
    | bool
    | None
    | list["ObjectType"]
    | dict[str, "ObjectType"]
    | tuple["ObjectType", ...]
)


class _Result[X](BaseModel):
    result: X = Field(..., description="The result", title="Result")


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


type MessageInput = str | File | Content | Sequence[str | File | Content]
"""Anything accepted by the `Message` constructor helpers."""


class Message(BaseModel):
    """A Responses API `message` item."""

    type: Literal["message"] = "message"
    id: str | None = None
    role: MessageRole
    content: Sequence[Content]
    status: ItemStatus | None = None

    @staticmethod
    def _coerce(content: MessageInput, output: bool) -> list[Content]:
        if isinstance(content, (str, File, BaseModel)):
            items: Sequence[Any] = [content]
        else:
            items = list(content)
        parts: list[Content] = []
        for c in items:
            if isinstance(c, str):
                parts.append(OutputText(text=c) if output else InputText(text=c))
            elif isinstance(c, File):
                if output:
                    raise ValueError(
                        "files are not valid content for an assistant message"
                    )
                parts.append(input_content_from_file(c))
            else:
                parts.append(c)
        return parts

    @classmethod
    def user(cls, content: MessageInput, *, id: str | None = None) -> Self:
        """Build a user message. Plain strings become `input_text`, `File`s become
        `input_image` / `input_audio` / `input_file` based on their media type."""
        return cls(role="user", content=cls._coerce(content, output=False), id=id)

    @classmethod
    def assistant(cls, content: MessageInput, *, id: str | None = None) -> Self:
        """Build an assistant message. Plain strings become `output_text`."""
        return cls(role="assistant", content=cls._coerce(content, output=True), id=id)

    @classmethod
    def system(cls, content: MessageInput, *, id: str | None = None) -> Self:
        return cls(role="system", content=cls._coerce(content, output=False), id=id)

    @classmethod
    def developer(cls, content: MessageInput, *, id: str | None = None) -> Self:
        return cls(role="developer", content=cls._coerce(content, output=False), id=id)

    @property
    def text(self) -> str:
        """All text content of the message, concatenated."""
        return "".join(
            c.text for c in self.content if isinstance(c, (InputText, OutputText))
        )

    @property
    def refusal(self) -> str | None:
        """The refusal text, if the model refused."""
        parts = [c.refusal for c in self.content if isinstance(c, Refusal)]
        return "".join(parts) if parts else None

    def parse[T: ObjectType](self, return_type: type[T]) -> T:
        """Parse the message text as JSON matching `return_type`."""

        class Result(_Result[return_type]): ...

        if inspect.isclass(return_type) and issubclass(return_type, BaseModel):
            response_format = return_type
        else:
            response_format = Result
        result = response_format.model_validate_json(self.text)
        if isinstance(result, _Result):
            return result.result
        return result

    def to_responses_item(self) -> dict[str, Any]:
        item: dict[str, Any] = {
            "type": "message",
            "role": self.role,
            "content": [c.model_dump(exclude_none=True) for c in self.content],
        }
        return item


class FunctionCall(BaseModel):
    """A Responses API `function_call` item."""

    type: Literal["function_call"] = "function_call"
    id: str | None = None
    call_id: str
    name: str
    arguments: str = "{}"
    status: ItemStatus | None = None

    @classmethod
    def create(
        cls,
        *,
        call_id: str,
        name: str,
        input: dict[str, JsonValue] | None = None,
        id: str | None = None,
    ) -> Self:
        return cls(call_id=call_id, name=name, arguments=json.dumps(input or {}), id=id)

    @property
    def input(self) -> dict[str, JsonValue]:
        """The parsed `arguments` JSON object."""
        if not self.arguments:
            return {}
        parsed = json.loads(self.arguments)
        if not isinstance(parsed, dict):
            raise ValueError(f"tool call arguments are not an object: {self.arguments}")
        return parsed

    def to_responses_item(self) -> dict[str, Any]:
        item: dict[str, Any] = {
            "type": "function_call",
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
        }
        if self.id:
            item["id"] = self.id
        return item


class FunctionCallOutput(BaseModel):
    """A Responses API `function_call_output` item."""

    type: Literal["function_call_output"] = "function_call_output"
    id: str | None = None
    call_id: str
    output: str | Sequence[InputContent] = ""
    status: ItemStatus | None = None

    name: str | None = None
    """The tool this output came from. Not sent to the provider; kept for callers."""

    is_error: bool | None = None
    """Whether the output is an error. Not sent to the provider; kept for callers."""

    @classmethod
    def create(
        cls,
        *,
        call_id: str,
        output: JsonValue,
        name: str | None = None,
        is_error: bool | None = None,
        files: Sequence[File] | None = None,
        id: str | None = None,
    ) -> Self:
        """Build an output item, JSON-encoding structured output and attaching files."""
        if files:
            output = {
                "output": output,
                "files_hint": f"The tool output includes {len(files)} file(s).",
                "files": [f.id for f in files],
            }
        text = json.dumps(output) if isinstance(output, (list, dict)) else str(output)
        content: str | Sequence[InputContent]
        if files:
            content = [
                InputText(text=text),
                *[input_content_from_file(f) for f in files],
            ]
        else:
            content = text
        return cls(call_id=call_id, output=content, name=name, is_error=is_error, id=id)

    @property
    def text(self) -> str:
        """The textual part of the output."""
        if isinstance(self.output, str):
            return self.output
        return "".join(c.text for c in self.output if isinstance(c, InputText))

    def to_responses_item(self) -> dict[str, Any]:
        output: Any = (
            self.output
            if isinstance(self.output, str)
            else [c.model_dump(exclude_none=True) for c in self.output]
        )
        item: dict[str, Any] = {
            "type": "function_call_output",
            "call_id": self.call_id,
            "output": output,
        }
        return item


class Reasoning(BaseModel):
    """A Responses API `reasoning` item.

    Responses are never stored server-side, so an item can only be replayed if it
    keeps its provider `id` and `encrypted_content`.
    """

    type: Literal["reasoning"] = "reasoning"
    id: str | None = None
    content: Sequence[ReasoningText] = Field(default_factory=list)
    summary: Sequence[SummaryText] = Field(default_factory=list)
    encrypted_content: str | None = None
    status: ItemStatus | None = None

    signature: str | None = None
    """Provider signature over the reasoning content, used by some models to replay it."""

    format: str | None = None
    """Provider-specific reasoning format tag, e.g. `openai-responses-v1`."""

    @classmethod
    def create(
        cls,
        text: str = "",
        *,
        summary: Sequence[str] = (),
        id: str | None = None,
        encrypted_content: str | None = None,
        signature: str | None = None,
        format: str | None = None,
    ) -> Self:
        return cls(
            content=[ReasoningText(text=text)] if text else [],
            summary=[SummaryText(text=s) for s in summary],
            id=id,
            encrypted_content=encrypted_content,
            signature=signature,
            format=format,
        )

    @property
    def text(self) -> str:
        """The raw reasoning text, from the `reasoning_text` content blocks."""
        return "".join(c.text for c in self.content)

    @property
    def summary_text(self) -> str:
        """The reasoning summary, from the `summary_text` blocks."""
        return "\n\n".join(s.text for s in self.summary if s.text)

    @property
    def display_text(self) -> str:
        """The human-readable reasoning: the raw text if present, else the summary."""
        return self.text or self.summary_text

    def to_responses_item(self) -> dict[str, Any] | None:
        """Serialize, or return `None` if the item cannot be replayed."""
        if not self.id:
            return None
        item: dict[str, Any] = {
            "type": "reasoning",
            "id": self.id,
            "summary": [s.model_dump() for s in self.summary],
        }
        if self.content:
            item["content"] = [c.model_dump() for c in self.content]
        if self.encrypted_content:
            item["encrypted_content"] = self.encrypted_content
        if self.signature:
            item["signature"] = self.signature
        if self.format:
            item["format"] = self.format
        return item


class Compaction(BaseModel):
    """A compacted run of earlier items.

    The Responses API defines `compaction` as an opaque, server-held item. Agentia
    compacts client-side instead, so the summary is plain text and the original items
    are kept for inspection; it serializes as an assistant message.
    """

    type: Literal["compaction"] = "compaction"
    id: str = Field(default_factory=lambda: _gen_id("cmp"))
    summary: str
    original_items: list["Item"] = Field(default_factory=list)
    encrypted_content: str | None = None
    created_by: str | None = None

    def flatten(self) -> list["Item"]:
        """The original items, with nested compactions expanded."""
        items: list[Item] = []
        for item in self.original_items:
            if isinstance(item, Compaction):
                items.extend(item.flatten())
            else:
                items.append(item)
        return items

    def to_responses_item(self) -> dict[str, Any]:
        return {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "[[COMPACTED CONVERSATION SUMMARY]]",
                    "annotations": [],
                },
                {"type": "output_text", "text": self.summary, "annotations": []},
                {
                    "type": "output_text",
                    "text": "[[END OF CONVERSATION SUMMARY]]",
                    "annotations": [],
                },
            ],
        }


type Item = Annotated[
    Message | FunctionCall | FunctionCallOutput | Reasoning | Compaction,
    Field(discriminator="type"),
]

type InputItem = str | Item | Sequence[Item]
"""Anything accepted where a prompt or list of items is expected."""


Compaction.model_rebuild()


def to_responses_input(items: Sequence[Item]) -> list[dict[str, Any]]:
    """Flatten items into a Responses API `input` array, dropping unreplayable ones."""
    out: list[dict[str, Any]] = []
    for item in items:
        if serialized := item.to_responses_item():
            out.append(serialized)
    return out


__all__ = [
    "ItemStatus",
    "MessageRole",
    "ObjectType",
    "MessageInput",
    "Message",
    "FunctionCall",
    "FunctionCallOutput",
    "Reasoning",
    "Compaction",
    "Item",
    "InputItem",
    "to_responses_input",
]
