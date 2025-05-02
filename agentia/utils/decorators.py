import asyncio
from enum import Enum, StrEnum
import inspect
import json
import logging
import types
from typing import (
    Any,
    Callable,
    Literal,
    TypeVar,
    Coroutine,
    Optional,
    Union,
    overload,
    get_args,
    get_origin,
    Annotated,
    TYPE_CHECKING,
)
from pydantic import BaseModel, Field
from openai.lib._parsing._completions import to_strict_json_schema  # type: ignore

if TYPE_CHECKING:
    from agentia.tools import Tools


class ToolFuncParam:
    def __init__(self, param: inspect.Parameter, fname: str, enum_check=True) -> None:
        assert isinstance(param, inspect.Parameter)
        self.func_name = fname
        self.param = param
        self.name = param.name
        self.description = self.__get_desc()
        t, r = self.__get_type()
        self.type = t
        self.required = r
        self.default = self.__get_default()
        self.type_name = self.__get_type_name(t)
        self.is_self = self.name == "self"
        self.enum = self.__get_enum(t, enum_check)

    def get_json_schema(self) -> dict[str, Any]:
        if issubclass(self.type, BaseModel):
            prop = to_strict_json_schema(self.type)
        else:
            prop = {}
            prop["type"] = self.type_name
            if self.enum is not None:
                prop["enum"] = self.enum
            if self.description:
                prop["description"] = self.description
        return prop

    def __get_desc(self) -> str | None:
        meta = (
            get_args(self.param.annotation)[1]
            if get_origin(self.param.annotation) == Annotated
            else None
        )
        return meta if isinstance(meta, str) else None

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

    def __get_type_name(self, t: type) -> str:
        match t:
            # string type
            case x if x == str:
                return "string"
            # integer type
            case x if x == int:
                return "integer"
            # boolean type
            case x if x == bool:
                return "boolean"
            # string enum
            case x if get_origin(x) == Annotated and get_args(x)[0] == str:
                return "string"
            case x if get_origin(x) == Literal:
                return "string"
            case x if issubclass(x, StrEnum) or issubclass(x, Enum):
                return "string"
            case x if issubclass(x, BaseModel):
                return "object"
            case _other:
                assert (
                    False
                ), f"Invalid type annotation for parameter `{self.param.name}` in function {self.func_name}"

    def __get_enum(self, t: type, check: bool) -> list[str] | None:
        match t:
            # string enum
            case x if get_origin(x) == Annotated and get_args(x)[0] == str:
                args = get_args(x)[1]
            case x if get_origin(x) == Literal:
                args = get_args(x)
            case x if issubclass(x, StrEnum) or issubclass(x, Enum):
                args = [item.value for item in x]
            case _:
                return None
        if check:
            for arg in args:
                if not isinstance(arg, str):
                    raise ValueError(
                        f"{self.func_name}.{self.name}: Literal members must be strings only"
                    )
        return [str(x) for x in args]


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


R2 = TypeVar("R2")


def _gen_prompt(f: Callable[..., R2], args: list[Any], kwargs: dict[str, Any]) -> str:
    desc = "You need to do the following task and return the result in JSON:\n\n"
    desc += "TASK NAME: " + f.__name__ + "\n"
    if doc := f.__doc__:
        desc += "DESCRIPTION: \n" + doc + "\n\n"
    else:
        logging.getLogger("agentia").warning(
            f"WARNING: Function {f.__name__} has no docstring. It's recommended to use the docstring to provide the agent with a description of the task."
        )
    sig = inspect.signature(f)
    if len(sig.parameters) > 0:
        params = [
            ToolFuncParam(p, f.__name__, enum_check=False)
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
            pdesc = " * " + p.name + ": "
            if isinstance(value, BaseModel):
                value_str = value.model_dump_json()
            else:
                value_str = json.dumps(value)
            if p.description or p.enum:
                pdesc += "\n"
                if s := p.description:
                    pdesc += f"    * description: {s}\n"
                if s := p.enum:
                    pdesc += f"    * possible values: {', '.join(s)}\n"
                pdesc += "    * value: " + value_str + "\n"
            else:
                pdesc += value_str + "\n"
            desc += pdesc
    return desc


class Result[T](BaseModel):
    result: T = Field(..., description="The result of the task", title="Task Result")


@overload
def magic(func: Callable[..., R2], /) -> Callable[..., R2]: ...


@overload
def magic(
    *,
    model: str | None = None,
    api_key: str | None = None,
    tools: Optional["Tools"] = None,
) -> Callable[[Callable[..., R2]], Callable[..., R2]]: ...


def magic(
    func: Callable[..., R2] | None = None,
    /,
    *,
    model: str | None = None,
    api_key: str | None = None,
    tools: Optional["Tools"] = None,
) -> Callable[..., R2] | Callable[[Callable[..., R2]], Callable[..., R2]]:
    """
    Decorator to mark a function as an agent for the agent.
    """

    def __magic_impl(callable: Callable[..., R2]) -> Callable[..., R2]:
        async def __func_impl(*args: Any, **kwargs: Any):
            from agentia import Agent

            prompt = _gen_prompt(callable, list(args), kwargs)

            agent = Agent(model=model, api_key=api_key, tools=tools)
            return_type = inspect.signature(callable).return_annotation
            if isinstance(return_type, inspect._empty):
                return_type = str
            assert isinstance(return_type, type), "The return type must be a type"
            if issubclass(return_type, BaseModel):
                response_format = return_type
            else:

                class _Result(Result[return_type]): ...

                response_format = _Result
            run = await agent.run(prompt, response_format=response_format)
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
