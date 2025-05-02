from dataclasses import dataclass
import inspect
from inspect import Parameter
import json
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Sequence,
    TYPE_CHECKING,
    Type,
)

import rich

from agentia.agent import CommunicationEvent, ToolCallEvent, UserConsentEvent

from .message import JSON, FunctionCall, ToolCall, ToolMessage

from .plugins import Plugin, ToolResult
from pydantic import BaseModel

from agentia.utils.decorators import tool, ToolFuncParam
from openai.lib._parsing._completions import to_strict_json_schema  # type: ignore

Tool = Plugin | Callable[..., Any]

Tools = Sequence[Tool]

if TYPE_CHECKING:
    from .agent import Agent

NAME_TAG = "agentia_tool_name"
DISPLAY_NAME_TAG = "agentia_tool_display_name"
IS_TOOL_TAG = "agentia_tool_is_tool"
DESCRIPTION_TAG = "agentia_tool_description"
METADATA_TAG = "agentia_tool_metadata"


@dataclass
class ToolInfo:
    name: str
    display_name: str
    description: str
    parameters: dict[str, Any]
    callable: Callable[..., Any]
    metadata: Any | None = None

    def to_json(self) -> JSON:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ClientTool(BaseModel):
    name: str
    description: str
    properties: dict[str, object]


class ToolRegistry:
    def __init__(self, agent: "Agent", tools: Tools | None = None) -> None:
        self.__functions: dict[str, ToolInfo] = {}
        self.__plugin_of_function: dict[str, str] = {}
        self.__plugins: list[Plugin] = []
        self._agent = agent
        for t in tools or []:
            if inspect.isfunction(t):
                self.__add_function(t)
            elif isinstance(t, Plugin):
                self.__add_plugin(t)
            else:
                raise ValueError(f"Invalid tool type: {type(t)}")
        # names = ", ".join([f"{k}" for k in self.__functions.keys()])
        # self._agent.log.info(f"Registered Tools: {names or 'N/A'}")

    async def init(self, silent: bool):
        from .plugins import PluginInitError

        for plugin in self.__plugins:
            if not silent:
                rich.print(f"[bold blue]>[/bold blue] [blue]{plugin.id()}[/blue]")
            self._agent.log.info(f" - init plugin: {plugin.id()}")
            try:
                await plugin.init()
            except Exception as e:
                self._agent.log.error(e)
                if not silent:
                    rich.print(
                        f"[red bold]Failed to initialize plugin `{plugin.id()}`[/red bold][red]: {e}[/red]"
                    )
                raise PluginInitError(plugin.id(), e) from e

    def add_tool(self, f: Callable[..., Any]):
        return self.__add_function(f)

    def add_client_tools(self, tools: list[ClientTool]):
        async def call_client_tool(tool_name: str, **kwargs):
            tool = next(t for t in tools if t.name == tool_name)
            # return await self._agent._client_tool_call(tool_name, kwargs)
            raise NotImplementedError(
                "Client tools are not supported yet. Please use the plugin system."
            )

        for t in tools:
            if t.name in self.__functions:
                raise ValueError(f"Tool `{t.name}` already exists")
            self.__functions[t.name] = ToolInfo(
                name=t.name,
                display_name=t.name,
                description=t.description,
                parameters=t.properties,
                callable=call_client_tool,
                metadata=None,
            )

    def __add_function(self, f: Callable[..., Any]):
        fname = getattr(f, NAME_TAG, f.__name__)
        params: Any = {"type": "object", "properties": {}, "required": []}
        for pname, param in inspect.signature(f).parameters.items():
            # Skip self parameter
            if pname == "self":
                continue
            p = ToolFuncParam(param, fname)
            # Add the parameter to the properties
            if p.required:
                params["required"].append(pname)
            params["properties"][pname] = p.get_json_schema()

        tool_info = ToolInfo(
            name=fname,
            display_name=getattr(f, DISPLAY_NAME_TAG, fname),
            description=getattr(f, DESCRIPTION_TAG, f.__doc__) or "",
            parameters=params,
            callable=f,
            metadata=getattr(f, METADATA_TAG, None),
        )
        self.__functions[tool_info.name] = tool_info
        return tool_info

    def __add_plugin(self, p: Plugin):
        # Add all functions from the plugin
        plugin_name = p.name()
        for _n, method in inspect.getmembers(p, predicate=inspect.ismethod):
            if not getattr(method, IS_TOOL_TAG, False):
                continue
            tool_info = self.__add_function(method)
            if not tool_info.name.startswith(plugin_name + "__"):
                old_name = tool_info.name
                tool_info.name = plugin_name + "__" + tool_info.name
                if not hasattr(tool_info, DISPLAY_NAME_TAG):
                    tool_info.display_name = plugin_name + "@" + tool_info.display_name
                del self.__functions[old_name]
                self.__functions[tool_info.name] = tool_info
            self.__plugin_of_function[tool_info.name] = p.id()
        # Add the plugin to the list of plugins
        self.__plugins.append(p)
        # Call the plugin's register method
        p._register(self._agent)

    def get_plugin(self, type: Type[Plugin]) -> Plugin | None:
        for p in self.__plugins:
            if isinstance(p, type):
                return p
        return None

    def is_empty(self) -> bool:
        return len(self.__functions) == 0

    def to_json(self) -> list[JSON]:
        functions = [v.to_json() for (k, v) in self.__functions.items()]
        return [{"type": "function", "function": f} for f in functions]

    def __filter_args(self, callable: Callable[..., Any], args: Any):
        from .agent import Agent

        p_args: list[Any] = []
        kw_args: dict[str, Any] = {}
        for p in inspect.signature(callable).parameters.values():
            if p.name == "self":
                continue
            param = ToolFuncParam(p, callable.__name__)
            is_model = issubclass(param.type, BaseModel)

            def get_value(name: str, default: Any):
                if p.name not in args:
                    return default
                if is_model:
                    return param.type(**args[p.name])
                else:
                    return args[p.name]

            match p.kind:
                case Parameter.POSITIONAL_ONLY if p.annotation == Agent:
                    p_args.append(self._agent)
                case Parameter.POSITIONAL_ONLY:
                    default = p.default if p.default != Parameter.empty else None
                    p_args.append(get_value(p.name, default))
                case Parameter.POSITIONAL_OR_KEYWORD | Parameter.KEYWORD_ONLY if (
                    p.annotation == Agent
                ):
                    kw_args[p.name] = self._agent
                case Parameter.POSITIONAL_OR_KEYWORD | Parameter.KEYWORD_ONLY:
                    default = p.default if p.default != Parameter.empty else None
                    kw_args[p.name] = get_value(p.name, default)
                case other:
                    raise ValueError(f"{other} is not supported")
        return p_args, kw_args

    async def call_function(
        self, function_call: FunctionCall, tool_id: str | None
    ) -> AsyncGenerator[UserConsentEvent | CommunicationEvent, None]:
        name = function_call.name
        args = function_call.arguments
        args_s = ""
        if isinstance(args, dict):
            for k, v in args.items():
                args_s += f"{k}={repr(v)}, "
            args_s = args_s[:-2]
        else:
            args_s = str(args)
        self._agent.log.info(f"TOOL#{tool_id} {name} {args_s}")
        # key = func_name if not func_name.startswith("functions.") else func_name[10:]
        if name not in self.__functions:
            raise ToolResult({"error": f"Tool `{name}` not found"})
        func = self.__functions[name].callable
        args, kw_args = self.__filter_args(func, args)
        try:
            result_or_coroutine = func(*args, **kw_args)
            if inspect.isasyncgen(result_or_coroutine):
                try:
                    next_value = None
                    while True:
                        if next_value is not None:
                            yielded = await result_or_coroutine.asend(next_value)
                        else:
                            yielded = await anext(result_or_coroutine)
                        if isinstance(yielded, UserConsentEvent):
                            assert (
                                yielded.response is None
                            ), "Newly created user consent event should not have a response"
                            yielded.tool = name
                            yielded.plugin = self.__plugin_of_function.get(name)
                            self._agent.log.info(f"TOOL#{tool_id} {yielded}")
                            yield yielded
                            self._agent.log.info(f"TOOL#{tool_id} {yielded}")
                            next_value = yielded.response
                        elif isinstance(yielded, CommunicationEvent):
                            self._agent.log.info(f"TOOL#{tool_id} {yielded}")
                            yield yielded
                        elif isinstance(yielded, ToolResult):
                            result = yielded.result
                            break
                except StopAsyncIteration as e:
                    result = {}
                except ToolResult as e:
                    result = e.result
            elif inspect.isgenerator(result_or_coroutine):
                try:
                    next_value = None
                    while True:
                        if next_value is not None:
                            yielded = result_or_coroutine.send(next_value)
                        else:
                            yielded = next(result_or_coroutine)
                        next_value = None
                        if isinstance(yielded, UserConsentEvent):
                            assert (
                                yielded.response is None
                            ), "Newly created user consent event should not have a response"
                            self._agent.log.info(f"TOOL#{tool_id} {yielded}")
                            yield yielded
                            self._agent.log.info(f"TOOL#{tool_id} {yielded}")
                            next_value = yielded.response
                        elif isinstance(yielded, CommunicationEvent):
                            self._agent.log.info(f"TOOL#{tool_id} {yielded}")
                            yield yielded
                        elif isinstance(yielded, ToolResult):
                            result = yielded.result
                            break
                except StopIteration as e:
                    result = e.value
                except ToolResult as e:
                    result = e.result
            elif inspect.iscoroutine(result_or_coroutine):
                result = await result_or_coroutine
            else:
                result = result_or_coroutine
        except EOFError as e:
            raise e
        except Exception as e:
            # print(f"TOOL#{tool_id} {name} ERROR: <{e}> <{type(e)}>")
            self._agent.log.error(e)
            raise ToolResult({"error": f"Failed to run tool `{name}`: {e}"}) from e
        result_s = json.dumps(result)
        self._agent.log.info(f"TOOL#{tool_id} {name} -> {result_s}")
        raise ToolResult(result)

    async def call_tools(self, tool_calls: Sequence[ToolCall]):
        for t in tool_calls:
            assert t.type == "function"
            assert t.function.name in self.__functions
            name = t.function.name
            info = self.__functions[name]

            event = ToolCallEvent(
                id=t.id,
                agent=self._agent.id,
                name=info.name,
                display_name=info.display_name,
                description=info.description,
                arguments=t.function.arguments,  # type: ignore
                metadata=info.metadata,
            )
            yield event
            raw_result = {}
            try:
                async for e in self.call_function(t.function, tool_id=t.id):
                    yield e
            except ToolResult as e:
                raw_result = e.result
            event = ToolCallEvent(
                id=t.id,
                agent=self._agent.id,
                name=info.name,
                display_name=info.display_name,
                description=info.description,
                arguments=t.function.arguments,  # type: ignore
                result=raw_result if raw_result is not None else {},
                metadata=info.metadata,
            )
            yield event
            if not isinstance(raw_result, str):
                result = json.dumps(raw_result)
            else:
                result = raw_result
            if t.id is not None:
                result_msg = ToolMessage(tool_call_id=t.id, content=result)
            else:
                raise NotImplementedError("legacy functions not supported")
            yield result_msg


__all__ = ["Tool", "Tools", "ToolInfo", "ClientTool", "ToolRegistry"]
