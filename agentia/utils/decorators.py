import asyncio
from dataclasses import dataclass, is_dataclass
import inspect
import json
import logging
import types
from typing import (
    Any,
    Callable,
    TypeVar,
    Coroutine,
    Optional,
    Union,
    cast,
    overload,
    get_args,
    get_origin,
    Annotated,
    TYPE_CHECKING,
)
from pydantic import BaseModel, Field, TypeAdapter, ConfigDict
from openai.lib._pydantic import _ensure_strict_json_schema  # type: ignore
from PIL.Image import Image
import base64
from io import BytesIO

from agentia.message import UserMessage, ContentPartText, ContentPartImage

if TYPE_CHECKING:
    from agentia.tools import Tools


@dataclass
class ImageUrl:
    """
    A special class for magic functions to represent an image URL argument.
    """

    url: str


def _is_image_type(t: type) -> bool:
    return (t == (ImageUrl | Image)) or issubclass(t, Image) or issubclass(t, ImageUrl)


class ToolFuncParam:
    def __init__(self, param: inspect.Parameter, fname: str, is_magic=False) -> None:
        self.is_magic = is_magic
        self.func_name = fname
        self.param = param
        self.name = param.name
        t, r = self.__get_type()
        self.type = t
        self.required = r
        self.description = self.__get_desc()
        self.default = self.__get_default()
        self.is_self = self.name == "self"
        self.schema = self.__get_json_schema()

    def __get_json_schema(self) -> dict[str, Any] | None:
        if self.is_magic and _is_image_type(self.type):
            return None

        if issubclass(self.type, BaseModel):
            schema = self.type.model_json_schema()
        else:
            schema = TypeAdapter(self.type).json_schema()
            if self.description:
                schema["description"] = self.description
        return _ensure_strict_json_schema(schema, path=(), root=schema)

    def __get_desc(self) -> str | None:
        # if type has a docstring, add to description
        type_desc = None
        if is_dataclass(self.type) or isinstance(self.type, BaseModel):
            if desc := self.type.__doc__:
                type_desc = desc
        meta = (
            get_args(self.param.annotation)[1]
            if get_origin(self.param.annotation) == Annotated
            else None
        )
        if meta is not None and isinstance(meta, str):
            if type_desc:
                return type_desc + "\n\n" + meta
            return meta
        if type_desc:
            return type_desc
        return None

    def __get_default(self) -> Any | None:
        if self.param.default == inspect.Parameter.empty:
            return None
        else:
            return self.param.default

    def __get_type(self) -> tuple[type, bool]:
        t = self.param.annotation
        t = t if get_origin(t) != Annotated else get_args(t)[0]
        if t == inspect.Parameter.empty:
            t = str  # the default type is string
        # Get parameter optionality
        param_t_is_opt = False
        is_optional = lambda x: (
            (get_origin(x) is Union or get_origin(x) is types.UnionType)
            and len(get_args(x)) == 2
            and type(None) in get_args(x)
        )
        if is_optional(t):
            t, param_t_is_opt = get_args(t)[0], True
        param_default_is_empty = self.param.default == inspect.Parameter.empty
        required = not param_t_is_opt and param_default_is_empty
        # Get parameter type
        assert not is_optional(t), "Optional types are not supported"
        return t, required


R = TypeVar("R", Coroutine[Any, Any, Optional[Any | str]], Optional[Any | str])


@overload
def tool(func: Callable[..., R], /) -> Callable[..., R]:
    """
    Decorator to mark a function as a tool for the agent.
    """
    ...


@overload
def tool(
    *,
    name: str | None = None,
    display_name: str | None = None,
    description: str | None = None,
    metadata: Any | None = None,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator to mark a function as a tool for the agent.
    """
    ...


def tool(
    func: Callable[..., R] | None = None,
    /,
    *,
    name: str | None = None,
    display_name: str | None = None,
    description: str | None = None,
    metadata: Any | None = None,
) -> Callable[..., R] | Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator to mark a function as a tool for the agent.
    """

    def __tool_impl(callable: Callable[..., R]) -> Callable[..., R]:
        from agentia.tools import (
            NAME_TAG,
            DISPLAY_NAME_TAG,
            DESCRIPTION_TAG,
            METADATA_TAG,
            IS_TOOL_TAG,
        )

        # store gpt function metadata to the callable object
        if isinstance(name, str):
            setattr(callable, NAME_TAG, name)
        if isinstance(display_name, str):
            setattr(callable, DISPLAY_NAME_TAG, display_name)
        if isinstance(description, str):
            setattr(callable, DESCRIPTION_TAG, description)
        if metadata is not None:
            setattr(callable, METADATA_TAG, metadata)
        setattr(callable, IS_TOOL_TAG, True)
        return callable

    if func is not None:
        return __tool_impl(func)

    return __tool_impl


F = TypeVar("F", bound=Callable)


def _gen_prompt(
    f: Callable, args: list[Any], kwargs: dict[str, Any]
) -> tuple[str, list[tuple[ToolFuncParam, Image | ImageUrl]]]:
    desc = "You need to do the following task and return the result in JSON:\n\n"
    desc += "TASK NAME: " + f.__name__ + "\n"
    if doc := f.__doc__:
        desc += "DESCRIPTION: \n" + doc + "\n\n"
    else:
        logging.getLogger("agentia").warning(
            f"WARNING: Function {f.__name__} has no docstring. It's recommended to use the docstring to provide the agent with a description of the task."
        )
    sig = inspect.signature(f)
    images: list[tuple[ToolFuncParam, Image | ImageUrl]] = []
    if len(sig.parameters) > 0:
        params = [
            ToolFuncParam(p, f.__name__, is_magic=True)
            for p in sig.parameters.values()
            if p.name != "self"
        ]
        params_with_values: dict[str, tuple[ToolFuncParam, Any]] = {}
        for a in args:
            # get and remove the first positional argument from params
            param = None
            for p in params:
                if p.param.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.POSITIONAL_ONLY,
                ):
                    params.remove(p)
                    param = p
                    break
            if param is None:
                raise ValueError("Too many positional arguments")
            else:
                params_with_values[param.name] = (param, a)
        for k, v in kwargs.items():
            # get and remove the first keyword argument from params
            param = None
            for p in params:
                if p.param.name != k:
                    continue
                if p.param.kind in (
                    inspect.Parameter.KEYWORD_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    params.remove(p)
                    param = p
                    break
            if param is None:
                raise ValueError("Unexpected keyword argument: " + k)
            else:
                params_with_values[param.name] = (param, v)
        for p in params:
            if p.name in params_with_values:
                continue
            if p.required:
                raise ValueError("Missing argument: " + p.name)
            params_with_values[p.name] = (p, p.default)
        desc += "USER'S INPUTS:\n\n"
        for p, value in params_with_values.values():
            if not p.schema:
                continue
            pdesc = " * " + p.name + ": "
            if isinstance(value, BaseModel):
                value_str = value.model_dump_json()
            else:
                value_str = json.dumps(value)
            if p.description or (p.schema and "enum" in p.schema):
                pdesc += "\n"
                if s := p.description:
                    pdesc += f"    * description: {s}\n"
                if s := (p.schema or {}).get("enum"):
                    pdesc += f"    * possible values: {', '.join(s)}\n"
                pdesc += "    * value: " + value_str + "\n"
            else:
                pdesc += value_str + "\n"
            desc += pdesc
        for p, value in params_with_values.values():
            if _is_image_type(p.type):
                value = cast(Image | ImageUrl, value)
                images.append((p, value))
    return desc, images


class Result[T](BaseModel):
    result: T = Field(..., description="The result of the task", title="Task Result")


@overload
def magic(func: F, /) -> F: ...


@overload
def magic(
    *,
    model: str | None = None,
    api_key: str | None = None,
    tools: Optional["Tools"] = None,
) -> Callable[[F], F]: ...


def magic(
    func: F | None = None,
    /,
    *,
    model: str | None = None,
    api_key: str | None = None,
    tools: Optional["Tools"] = None,
) -> F | Callable[[F], F]:
    """
    Decorator to mark a function as an agent for the agent.
    """

    def __magic_impl(callable: F) -> F:
        async def __func_impl(*args: Any, **kwargs: Any):
            from agentia import Agent

            prompt, images = _gen_prompt(callable, list(args), kwargs)

            agent = Agent(model=model, api_key=api_key, tools=tools)
            return_type = inspect.signature(callable).return_annotation
            if isinstance(return_type, inspect._empty):
                return_type = str
            if issubclass(return_type, BaseModel):
                response_format = return_type
            else:

                class _Result(Result[return_type]): ...

                response_format = _Result

            messages = [
                UserMessage(
                    content=prompt,
                    role="user",
                )
            ]
            for i, (p, image) in enumerate(images):
                if isinstance(image, ImageUrl):
                    url = image.url
                else:
                    buffered = BytesIO()
                    image.save(buffered, format=image.format)
                    img_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    match image.format:
                        case "JPEG":
                            content_type = "jpeg"
                        case "PNG":
                            content_type = "png"
                        case "GIF":
                            content_type = "gif"
                        case "BMP":
                            content_type = "bmp"
                        case "WEBP":
                            content_type = "webp"
                        case "ICO":
                            content_type = "ico"
                        case _:
                            raise ValueError(
                                f"Unsupported image format: {image.format}"
                            )
                    url = f"data:image/{content_type};base64,{img_data}"
                s = f"Image Argument #{i}: {p.name}"
                if p.description:
                    s += f"\nDescription: {p.description})"
                messages.append(
                    UserMessage(
                        content=[
                            ContentPartText(s),
                            ContentPartImage(url),
                        ],
                        role="user",
                    )
                )
            run = await agent.run(messages, response_format=response_format)
            assert run.content
            json_result = json.loads(run.content)
            if issubclass(return_type, BaseModel):
                return return_type(**json_result)
            else:
                result = Result[return_type](**json_result)
                return result.result

        if not inspect.iscoroutinefunction(callable):

            def __func_impl_sync(*args: Any, **kwargs: Any):
                return asyncio.run(__func_impl(*args, **kwargs))

            return __func_impl_sync  # type: ignore

        return __func_impl  # type: ignore

    if func is not None:
        return __magic_impl(func)

    return __magic_impl
