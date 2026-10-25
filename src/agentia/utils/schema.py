"""JSON schema helpers for tool parameters and structured output."""

import copy
from typing import Any

from pydantic import BaseModel


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline `$ref`s to `$defs` (non-recursive schemas only) and drop `$defs`."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def walk(x: Any, depth: int = 0) -> Any:
        if depth > 50:
            raise ValueError("schema is too deeply nested (recursive $ref?)")
        if isinstance(x, list):
            return [walk(v, depth + 1) for v in x]
        if isinstance(x, dict):
            if "$ref" in x and isinstance(x["$ref"], str):
                name = x["$ref"].rsplit("/", 1)[-1]
                target = walk(copy.deepcopy(defs[name]), depth + 1)
                rest = {k: walk(v, depth + 1) for k, v in x.items() if k != "$ref"}
                return {**target, **rest}
            return {k: walk(v, depth + 1) for k, v in x.items()}
        return x

    return walk(schema)


def strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a schema valid for strict function calling / structured output.

    Every object gets `additionalProperties: false` and lists all its properties as
    `required`; `$ref`s are inlined; `default: null` and single-element `allOf`s go.
    """

    def walk(x: Any) -> Any:
        if isinstance(x, list):
            return [walk(v) for v in x]
        if not isinstance(x, dict):
            return x
        x = {k: walk(v) for k, v in x.items()}
        if x.get("default", 0) is None:
            del x["default"]
        all_of = x.get("allOf")
        if isinstance(all_of, list) and len(all_of) == 1:
            del x["allOf"]
            x = {**all_of[0], **x}
        if x.get("type") == "object" or "properties" in x:
            props = x.get("properties") or {}
            x["properties"] = props
            x["required"] = list(props)
            x["additionalProperties"] = False
        return x

    return walk(inline_refs(schema))


def compact_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """A pydantic schema as a compact function-calling schema: refs inlined, no titles
    or defaults, `Optional[X]` → `X`, and objects closed with `additionalProperties`.

    Unlike `strict_json_schema`, optional properties stay optional."""

    def walk(x: Any) -> Any:
        if isinstance(x, list):
            return [walk(v) for v in x]
        if not isinstance(x, dict):
            return x
        out = {k: walk(v) for k, v in x.items() if k not in ("title", "default")}
        any_of = out.get("anyOf")
        if isinstance(any_of, list):
            non_null = [v for v in any_of if v != {"type": "null"}]
            if len(non_null) == 1 and len(non_null) < len(any_of):
                del out["anyOf"]
                out = {**non_null[0], **out}
        if out.get("type") == "object" and "properties" in out:
            out.setdefault("required", [])
            out["additionalProperties"] = False
        return out

    return walk(inline_refs(schema))


def schema_of_model(model: type[BaseModel]) -> dict[str, Any]:
    """The compact function-calling schema of a pydantic model."""
    return compact_json_schema(model.model_json_schema())


__all__ = [
    "compact_json_schema",
    "inline_refs",
    "schema_of_model",
    "strict_json_schema",
]
