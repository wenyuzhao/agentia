"""Shared primitives: usage, tool definitions, annotations and file payloads.

Responses API types
-------------------
- `Usage`               -- the `response.usage` object, flattened. `reasoning_tokens`
                           comes from `output_tokens_details.reasoning_tokens` and
                           `cached_input_tokens` from `input_tokens_details.cached_tokens`.
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
                           `response.status` plus `incomplete_details.reason`.
- `ToolSchema`          -- the union of the two tool definition types above.
- `UserConsentRequest`  -- agentia's human-in-the-loop consent prompt.
"""

from pathlib import Path
from typing import Annotated, Any, Literal
from pydantic import BaseModel, Field, JsonValue
from uuid import uuid4
import base64


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
]


type ToolChoice = Literal["auto", "none", "required"] | str


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None

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
                header, b64data = self.data.split(",", 1)
                return base64.b64decode(b64data)
            elif self.data.startswith(("http://", "https://")):
                raise ValueError("Cannot convert URL data to bytes")
            else:
                path = Path(self.data)
                if not path.is_file():
                    raise ValueError(f"File path '{self.data}' does not exist")
                return path.read_bytes()
        else:
            raise ValueError("Invalid data type for File")


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


class UserConsentRequest(BaseModel):
    type: Literal["user-consent-request"] = "user-consent-request"
    id: str = Field(default_factory=lambda: str(uuid4()))
    message: str
    details: str | None = None

    def __init__(self, message: str, details: str | None = None):
        super().__init__(message=message, details=details)


__all__ = [
    "DataContent",
    "FinishReason",
    "ToolChoice",
    "Usage",
    "FunctionTool",
    "ProviderDefinedTool",
    "ToolSchema",
    "File",
    "UrlCitation",
    "Annotation",
    "UserConsentRequest",
]
