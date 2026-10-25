"""Response format options.

Responses API types
-------------------
None directly. These serialize into the API's `text.format` object, whose three shapes
are `{"type": "text"}`, `{"type": "json_object"}` and
`{"type": "json_schema", "name", "schema", "strict"}`.

Agentia extensions
------------------
- `ResponseFormatText`  -- plain text output; emits `{"type": "text"}`.
- `ResponseFormatJson`  -- JSON output. With no `json_schema` it emits `json_object`,
                           otherwise a strict `json_schema`. `from_model()` derives the
                           schema from a pydantic model or any `ObjectType`, wrapping
                           non-model types in a `{"result": ...}` envelope.
- `ResponseFormat`      -- the union of the two.
"""

import builtins
import inspect
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, JsonValue

from ..utils.schema import strict_json_schema
from .items import ObjectType, _Result


class ResponseFormatText(BaseModel):
    type: Literal["text"] = "text"

    def to_responses_text_format(self) -> dict[str, Any]:
        return {"type": "text"}


class ResponseFormatJson(BaseModel):
    type: Literal["json"] = "json"
    json_schema: JsonValue | None = None
    name: str | None = None
    description: str | None = None

    @staticmethod
    def from_model[T: ObjectType](
        return_type: builtins.type[T],
    ) -> "ResponseFormatJson":
        class Result(_Result[return_type]): ...

        if inspect.isclass(return_type) and issubclass(return_type, BaseModel):
            response_format = return_type
        else:
            response_format = Result
        return ResponseFormatJson(
            json_schema=strict_json_schema(response_format.model_json_schema()),
            name=return_type.__name__,
            description=f"JSON object matching the schema of {return_type.__name__}",
        )

    def to_responses_text_format(self) -> dict[str, Any]:
        if not self.json_schema:
            return {"type": "json_object"}
        fmt: dict[str, Any] = {
            "type": "json_schema",
            "name": self.name or "response",
            "schema": self.json_schema,
            "strict": True,
        }
        if self.description:
            fmt["description"] = self.description
        return fmt


type ResponseFormat = Annotated[
    ResponseFormatText | ResponseFormatJson, Field(discriminator="type")
]


__all__ = [
    "ResponseFormat",
    "ResponseFormatJson",
    "ResponseFormatText",
]
