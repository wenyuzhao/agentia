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

from agentia.agent import CommunicationEvent, ToolCallEvent, UserConsentEvent, Event

from .message import JSON, ClientToolCallEvent, FunctionCall, ToolCall, ToolMessage

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


class ClientTool(BaseModel):
    name: str
    description: str
    properties: dict[str, object]
    metadata: Any | None = None


class ToolInfo:
    def __init__(
        self,
        f: Callable[..., Any],
        plugin: Plugin | None = None,
        client_tool: ClientTool | None = None,
    ):
        self.plugin = plugin
        self.func = f
        if client_tool:
            self.name = client_tool.name
            self.display_name = client_tool.name
            self.description = client_tool.description
            self.schema = {
                "name": client_tool.name,
                "description": client_tool.description or "",
                "parameters": client_tool.properties,
            }
            self.metadata = client_tool.metadata
            self.params = None
            self.client_tool = client_tool
        else:
            if plugin:
                self.name = (
                    getattr(f, NAME_TAG, None) or f"{plugin.name()}__{f.__name__}"
                )
                self.display_name: str = (
                    getattr(f, DISPLAY_NAME_TAG, None)
                    or f"{plugin.name()}@{f.__name__}"
                )
            else:
                self.name = getattr(f, NAME_TAG, f.__name__)
                self.display_name: str = getattr(f, DISPLAY_NAME_TAG, self.name)
            self.description: str = getattr(f, DESCRIPTION_TAG, f.__doc__) or ""
            self.metadata = getattr(f, METADATA_TAG, None)
            self.params = [
                ToolFuncParam(p, self.func.__name__)
                for p in inspect.signature(f).parameters.values()
                if p.name != "self"
            ]
            self.schema = self.get_json_schema(self.params)
            self.client_tool = None

    def get_json_schema(self, fparams: list[ToolFuncParam]) -> Any:
        params: Any = {"type": "object", "properties": {}, "required": []}
        for p in fparams:
            if p.required:
                params["required"].append(p.name)
            params["properties"][p.name] = p.schema
        return {
            "name": self.name,
            "description": self.description or "",
            "parameters": params,
        }

    def process_json_args(self, agent: "Agent", args: Any):
        from .agent import Agent

        p_args: list[Any] = []
        kw_args: dict[str, Any] = {}

        def get_value(name: str, default: Any, is_model: bool):
            if name not in args:
                return default
            if is_model:
                return p.type(**args[name])
            else:
                return args[name]

        assert self.params is not None
        for p in self.params:
            is_model = issubclass(p.type, BaseModel)
            match p.param.kind:
                case Parameter.POSITIONAL_ONLY if p.param.annotation == Agent:
                    p_args.append(agent)
                case Parameter.POSITIONAL_ONLY:
                    default = p.default if p.default != Parameter.empty else None
                    p_args.append(get_value(p.name, default, is_model))
                case Parameter.POSITIONAL_OR_KEYWORD | Parameter.KEYWORD_ONLY if (
                    p.param.annotation == Agent
                ):
                    kw_args[p.name] = agent
                case Parameter.POSITIONAL_OR_KEYWORD | Parameter.KEYWORD_ONLY:
                    default = p.default if p.default != Parameter.empty else None
                    kw_args[p.name] = get_value(p.name, default, is_model)
                case other:
                    raise ValueError(f"{other} is not supported")
        return p_args, kw_args


class ToolRegistry:
    def __init__(self, agent: "Agent", tools: Tools | None = None) -> None:
        self.__functions: dict[str, ToolInfo] = {}
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
            self.__functions[t.name] = ToolInfo(call_client_tool, client_tool=t)

    def __add_function(
        self,
        f: Callable[..., Any],
        plugin: Plugin | None = None,
    ):
        tool_info = ToolInfo(f, plugin=plugin)
        self.__functions[tool_info.name] = tool_info

    def __add_plugin(self, p: Plugin):
        # Add all functions from the plugin
        for _n, method in inspect.getmembers(p, predicate=inspect.ismethod):
            if not getattr(method, IS_TOOL_TAG, False):
                continue
            self.__add_function(method, plugin=p)
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

    def get_schema(self) -> list[Any]:
        functions = [v.schema for (k, v) in self.__functions.items()]
        return [{"type": "function", "function": f} for f in functions]

    async def call_function(
        self, function_call: FunctionCall, tool_id: str | None
    ) -> AsyncGenerator[Event, None]:
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
        func = self.__functions[name].func
        if t := self.__functions[name].client_tool:
            args2: dict[str, Any] = args  # type: ignore
            e = ClientToolCallEvent(tool=t, args=args2)
            result = yield e
            raise NotImplementedError("Client tools are not supported yet.")
            return
        args, kw_args = self.__functions[name].process_json_args(self._agent, args)
        try:
            result = func(*args, **kw_args)
            if inspect.isasyncgen(result) or inspect.isgenerator(result):
                try:
                    next_value = None
                    while True:
                        if inspect.isasyncgen(result):
                            if next_value is not None:
                                yielded = await result.asend(next_value)
                            else:
                                yielded = await anext(result)
                        elif inspect.isgenerator(result):
                            if next_value is not None:
                                yielded = result.send(next_value)
                            else:
                                yielded = next(result)
                        else:
                            assert False, "unreachable"
                        next_value = None
                        if isinstance(yielded, UserConsentEvent):
                            assert (
                                yielded.response is None
                            ), "Newly created user consent event should not have a response"
                            yielded.tool = name
                            if p := self.__functions[name].plugin:
                                yielded.plugin = p.id()
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
                except StopIteration as e:
                    result = e.value
                except ToolResult as e:
                    result = e.result
            elif inspect.iscoroutine(result):
                result = await result
            else:
                result = result
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
