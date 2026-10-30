from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, Field, JsonValue, model_validator
from uuid import uuid4
import base64
import json
import mimetypes


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


type ToolSchema = FunctionTool


class ToolCallPart(BaseModel):
    """A tool call requested by the model. Part of an `AssistantMessage`."""

    type: Literal["tool-call"] = "tool-call"

    tool_call_id: str
    """The identifier of the tool call. It must be unique across all tool calls."""

    tool_name: str
    """The name of the tool that should be called."""

    input: dict[str, JsonValue]
    """JSON object with the tool call arguments. Must match the parameters schema of the tool."""

    def __str__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.input.items())
        return f"→ {self.tool_name}({args})"


def _infer_media_type(data: DataContent) -> str | None:
    if isinstance(data, bytes):
        return None
    if data.startswith("data:"):
        header = data[len("data:") :].split(",", 1)[0]
        return header.split(";", 1)[0] or None
    path = data.split("?", 1)[0].split("#", 1)[0]
    media_type, _ = mimetypes.guess_type(path)
    return media_type


class File(BaseModel):
    """
    A file: an image, audio clip, video, PDF, etc.

    `File` is also the file part of a message, so it can be put directly into the
    content of a `UserMessage` or `AssistantMessage`:

        UserMessage(["What is in this picture?", File("cat.png")])

    `data` is raw bytes, a local path, an http(s) URL, a data URL, or a provider file
    ID (`file-...`). `media_type` is inferred from the path, URL or data URL when
    omitted; it is required for raw bytes.
    """

    type: Literal["file"] = "file"
    data: DataContent
    media_type: str
    filename: str | None = None
    id: str = Field(default_factory=lambda: str(uuid4()))

    def __init__(
        self, data: DataContent, media_type: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(data=data, media_type=media_type, **kwargs)

    @model_validator(mode="before")
    @classmethod
    def _fill_media_type(cls, values: Any) -> Any:
        if isinstance(values, dict) and not values.get("media_type"):
            data = values.get("data")
            media_type = (
                _infer_media_type(data) if isinstance(data, (bytes, str)) else None
            )
            if media_type is None:
                raise ValueError(
                    "Cannot infer the media type of the file. Please specify `media_type`."
                )
            values = {**values, "media_type": media_type}
        return values

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

    def to_base64(self) -> str:
        """The raw file content, base64 encoded (without a `data:` prefix)."""
        if isinstance(self.data, str) and self.data.startswith("data:"):
            return self.data.split(",", 1)[1]
        return base64.b64encode(self.to_bytes()).decode()

    def __str__(self) -> str:
        name = self.filename
        if (
            not name
            and isinstance(self.data, str)
            and not self.data.startswith("data:")
        ):
            name = self.data
        return f"[{self.media_type}: {name}]" if name else f"[{self.media_type}]"


class ToolResultPart(BaseModel):
    """The result of a tool call. Part of a `ToolMessage`."""

    type: Literal["tool-result"] = "tool-result"

    tool_call_id: str
    """The ID of the tool call that this result is associated with."""

    tool_name: str
    """Name of the tool that generated this result."""

    input: dict[str, JsonValue]
    """The arguments the tool was called with."""

    output: JsonValue
    """Result of the tool call. This is a JSON-serializable object."""

    output_files: list[File] = Field(default_factory=list)
    """Optional list of files generated by the tool call."""

    is_error: bool = False
    """Whether the tool call failed. `output` then describes the error."""

    def serialize_output(self) -> str:
        """The output as sent to the model: strings verbatim, anything else as JSON."""
        output = self.output
        if self.output_files:
            output = {
                "output": output,
                "files_hint": f"The tool output includes {len(self.output_files)} file(s).",
                "files": [f.id for f in self.output_files],
            }
        return output if isinstance(output, str) else json.dumps(output)

    def __str__(self) -> str:
        mark = "✗" if self.is_error else "←"
        output = (
            self.output if isinstance(self.output, str) else json.dumps(self.output)
        )
        return f"{mark} {self.tool_name}: {output}"


class UrlCitation(BaseModel):
    """A URL citation in the text of an assistant message."""

    type: Literal["url-citation"] = "url-citation"

    end: int
    """The index of the last character of the URL citation in the text."""

    start: int
    """The index of the first character of the URL citation in the text."""

    title: str
    """The title of the web resource."""

    url: str
    """The URL of the web resource."""


class UserConsentRequest(BaseModel):
    type: Literal["user-consent-request"] = "user-consent-request"
    id: str = Field(default_factory=lambda: str(uuid4()))
    message: str
    details: str | None = None

    def __init__(self, message: str, details: str | None = None):
        super().__init__(message=message, details=details)


# Aliases for the previous names
ToolCall = ToolCallPart
ToolCallResponse = ToolResultPart
Annotation = UrlCitation

__all__ = [
    "DataContent",
    "FinishReason",
    "ToolChoice",
    "Usage",
    "FunctionTool",
    "ToolSchema",
    "ToolCallPart",
    "ToolResultPart",
    "ToolCall",
    "ToolCallResponse",
    "File",
    "UrlCitation",
    "Annotation",
    "UserConsentRequest",
]
