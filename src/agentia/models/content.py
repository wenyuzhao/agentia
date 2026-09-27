"""Content parts of a `message` or `reasoning` item.

Responses API types
-------------------
- `InputText`      -- `input_text`, in user/system/developer messages.
- `InputImage`     -- `input_image`, with `image_url` or `file_id`, plus `detail`.
- `InputFile`      -- `input_file`, with `file_id`, `file_url`, or `file_data`.
- `InputAudio`     -- `input_audio`, wrapping `InputAudioData` (`data` + `format`).
- `OutputText`     -- `output_text`, in assistant messages, with `annotations`.
- `Refusal`        -- `refusal`, when the model declines.
- `ReasoningText`  -- `reasoning_text`, the raw reasoning inside a `reasoning` item.
- `SummaryText`    -- `summary_text`, a reasoning summary block.

Agentia extensions
------------------
- `InputContent` / `OutputContent` / `Content` / `ReasoningContent` -- discriminated
  unions over the types above; the API only implies them.
- `from_file()` classmethods and `input_content_from_file()` -- build the right content
  part from a `File`, routing by media type: images to `input_image`, audio to
  `input_audio`, and everything else (including video, which the API has no content type
  for) to `input_file`.
"""

from typing import Annotated, Any, Literal, Self
from pydantic import BaseModel, Field

from .base import Annotation, File


class InputText(BaseModel):
    type: Literal["input_text"] = "input_text"
    text: str

    def __init__(self, text: str = "", **data: Any) -> None:
        super().__init__(text=text, **data)


class InputImage(BaseModel):
    type: Literal["input_image"] = "input_image"
    image_url: str | None = None
    file_id: str | None = None
    detail: Literal["low", "high", "auto"] = "auto"

    @classmethod
    def from_file(cls, file: File) -> Self:
        """Build an `input_image` from a `File`, resolving paths and bytes to a URL."""
        if isinstance(file.data, str) and file.data.startswith("file-"):
            return cls(file_id=file.data)
        return cls(image_url=file.to_url())


class InputFile(BaseModel):
    type: Literal["input_file"] = "input_file"
    file_id: str | None = None
    file_url: str | None = None
    file_data: str | None = None
    filename: str | None = None

    @classmethod
    def from_file(cls, file: File) -> Self:
        """Build an `input_file` from a `File`, resolving paths and bytes to a data URL."""
        if isinstance(file.data, str) and file.data.startswith("file-"):
            return cls(file_id=file.data)
        return cls(file_data=file.to_url(), filename=file.filename)


class InputAudioData(BaseModel):
    data: str
    format: Literal["wav", "mp3"]


class InputAudio(BaseModel):
    type: Literal["input_audio"] = "input_audio"
    input_audio: InputAudioData

    @classmethod
    def from_file(cls, file: File) -> Self:
        """Build an `input_audio` from a `File`."""
        if isinstance(file.data, str) and file.data.startswith(("http://", "https://")):
            raise ValueError("audio content does not support URLs")
        match file.media_type:
            case "audio/wav" | "audio/x-wav":
                format: Literal["wav", "mp3"] = "wav"
            case "audio/mpeg" | "audio/mp3":
                format = "mp3"
            case _:
                raise ValueError(f"unsupported audio media type {file.media_type}")
        data = file.data if isinstance(file.data, str) else file.data.decode()
        return cls(input_audio=InputAudioData(data=data, format=format))


class OutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[Annotation] = Field(default_factory=list)

    def __init__(self, text: str = "", **data: Any) -> None:
        super().__init__(text=text, **data)


class Refusal(BaseModel):
    type: Literal["refusal"] = "refusal"
    refusal: str


class ReasoningText(BaseModel):
    type: Literal["reasoning_text"] = "reasoning_text"
    text: str

    def __init__(self, text: str = "", **data: Any) -> None:
        super().__init__(text=text, **data)


class SummaryText(BaseModel):
    type: Literal["summary_text"] = "summary_text"
    text: str

    def __init__(self, text: str = "", **data: Any) -> None:
        super().__init__(text=text, **data)


type InputContent = Annotated[
    InputText | InputImage | InputFile | InputAudio, Field(discriminator="type")
]

type OutputContent = Annotated[OutputText | Refusal, Field(discriminator="type")]

type Content = Annotated[
    InputText | InputImage | InputFile | InputAudio | OutputText | Refusal,
    Field(discriminator="type"),
]

type ReasoningContent = Annotated[ReasoningText, Field(discriminator="type")]


def input_content_from_file(file: File) -> InputContent:
    """Pick the right input content type for a file, based on its media type."""
    if file.media_type.startswith("image/"):
        return InputImage.from_file(file)
    if file.media_type.startswith("audio/"):
        return InputAudio.from_file(file)
    return InputFile.from_file(file)


__all__ = [
    "InputText",
    "InputImage",
    "InputFile",
    "InputAudio",
    "InputAudioData",
    "OutputText",
    "Refusal",
    "ReasoningText",
    "SummaryText",
    "InputContent",
    "OutputContent",
    "Content",
    "ReasoningContent",
    "input_content_from_file",
]
