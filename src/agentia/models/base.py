"""Shared primitives: usage, tool definitions, annotations and file payloads.

Responses API types
-------------------
- `Usage`               -- the `response.usage` object, flattened. `reasoning_tokens`
                           comes from `output_tokens_details.reasoning_tokens`,
                           `cached_input_tokens` from `input_tokens_details.cached_tokens`
                           and `cost` from OpenRouter's `usage.cost` extension.
- `FunctionTool`        -- a `function` tool definition (`to_responses_tool()` emits the
                           flat `{type, name, description, parameters, strict}` shape).
- `ProviderDefinedTool` -- a built-in, provider-executed tool (`web_search`,
                           `file_search`, `code_interpreter`, ...), emitted as
                           `{type: <id>, **args}`.
- `UrlCitation`         -- a `url_citation` annotation on an `output_text` part.
- `Annotation`          -- the annotation union (currently only `url_citation`).
- `ToolChoice`          -- the `tool_choice` parameter: "auto" / "none" / "required",
                           or a tool name for `{type: "function", name: ...}`.

Agentia extensions
------------------
- `File`                -- a media payload (bytes, path, URL or `file-` id) that content
                           parts are built from. Not an API type.
- `DataContent`         -- the `bytes | str` payload alias used by `File`.
- `FinishReason`        -- a normalized reason across providers. The API instead reports
                           `response.status` plus `incomplete_details.reason`. A run adds
                           `suspended`, `stopped` and `max-steps`.
- `ToolSchema`          -- the union of the two tool definition types above.
- `Instruction`         -- one block of the system prompt (static or dynamic).
- `UserConsentRequest`  -- agentia's human-in-the-loop consent prompt.
"""

import base64
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, JsonValue

type DataContent = bytes | str

type FinishReason = Literal[
    "stop",
    "length",
    "content-filter",
    "tool-calls",
    "interrupted",
    "error",
    "other",
    "unknown",
    # Run-level reasons (agentia.run.completed only):
    "suspended",  # a tool suspended; resume with `agent.run(None)`
    "stopped",  # a hook raised `StopRun`
    "max-steps",  # `LLMOptions.max_steps` reached
]


type ToolChoice = Literal["auto", "none", "required"] | str


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None
    cost: float | None = None
    """Cost in USD, when the provider reports it (OpenRouter's `usage.cost`)."""

    def __add__(self, y: "Usage") -> "Usage":
        x = self.model_copy()
        for k in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "reasoning_tokens",
            "cached_input_tokens",
            "cost",
        ):
            if v := getattr(y, k):
                setattr(x, k, (getattr(x, k) or 0) + v)
        return x


class FunctionTool(BaseModel):
    type: Literal["function"] = "function"
    name: str
    input_schema: JsonValue
    description: str | None = None
    strict: bool = False
    """
    Whether the model must produce arguments that exactly match `input_schema`.

    Strict mode requires the schema to set `additionalProperties: false` and list every
    property in `required`, so it stays opt-in for schemas derived from plain functions.
    """

    def to_responses_tool(self) -> Any:
        """Serialize to a Responses API function tool definition."""
        tool: dict[str, Any] = {
            "type": "function",
            "name": self.name,
            "parameters": self.input_schema,
            "strict": self.strict,
        }
        if self.description is not None:
            tool["description"] = self.description
        return tool


class ProviderDefinedTool(BaseModel):
    type: Literal["provider-defined"] = "provider-defined"
    id: str
    name: str
    args: dict[str, JsonValue] = Field(default_factory=dict)

    def to_responses_tool(self) -> Any:
        """Serialize to a Responses API built-in tool definition."""
        return {"type": self.id, **self.args}


type ToolSchema = Annotated[
    FunctionTool | ProviderDefinedTool, Field(discriminator="type")
]


class File(BaseModel):
    media_type: str
    data: DataContent
    id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str | None = None

    def to_url(self) -> str:
        if isinstance(self.data, bytes):
            return (
                f"data:{self.media_type};base64,{base64.b64encode(self.data).decode()}"
            )
        elif isinstance(self.data, str):
            if self.data.startswith(("http://", "https://", "data:")):
                return self.data
            else:
                path = Path(self.data)
                if not path.is_file():
                    raise ValueError(f"File path '{self.data}' does not exist")
                return f"data:{self.media_type};base64,{base64.b64encode(path.read_bytes()).decode()}"
        else:
            return str(self.data)

    def to_bytes(self) -> bytes:
        if isinstance(self.data, bytes):
            return self.data
        elif isinstance(self.data, str):
            if self.data.startswith("data:"):
                _header, b64data = self.data.split(",", 1)
                return base64.b64decode(b64data)
            elif self.data.startswith(("http://", "https://")):
                raise ValueError("Cannot convert URL data to bytes")
            else:
                path = Path(self.data)
                if not path.is_file():
                    raise ValueError(f"File path '{self.data}' does not exist")
                return path.read_bytes()
        else:
            raise TypeError("Invalid data type for File")


class UrlCitation(BaseModel):
    type: Literal["url_citation"] = "url_citation"

    end_index: int = 0
    """The index of the last character of the citation in the message."""

    start_index: int = 0
    """The index of the first character of the citation in the message."""

    title: str = ""
    """The title of the web resource."""

    url: str
    """The URL of the web resource."""


type Annotation = Annotated[UrlCitation, Field(discriminator="type")]


class Instruction(BaseModel):
    """One block of the system prompt.

    Static blocks come first, then the compaction summary (if any), then dynamic blocks,
    so the stable prefix stays cacheable even when dynamic content (dates, todo lists)
    changes every step. Providers that take explicit cache breakpoints get one after the
    static blocks and one after the summary.
    """

    text: str
    dynamic: bool = False


class UserConsentRequest(BaseModel):
    type: Literal["user-consent-request"] = "user-consent-request"
    id: str = Field(default_factory=lambda: str(uuid4()))
    message: str
    details: str | None = None

    def __init__(self, message: str, details: str | None = None):
        super().__init__(message=message, details=details)


__all__ = [
    "Annotation",
    "DataContent",
    "File",
    "FinishReason",
    "FunctionTool",
    "Instruction",
    "ProviderDefinedTool",
    "ToolChoice",
    "ToolSchema",
    "UrlCitation",
    "Usage",
    "UserConsentRequest",
]
