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

from .message import FunctionCall, ToolCall, ToolMessage

from .plugins import Plugin, ToolResult
from pydantic import BaseModel

from agentia.utils.decorators import tool, ToolFuncParam
from openai.lib._parsing._completions import to_strict_json_schema  # type: ignore

from .mcp import MCPServer

Tool = Plugin | Callable[..., Any] | MCPServer

Tools = Sequence[Tool]

if TYPE_CHECKING:
    from .agent import Agent

NAME_TAG = "agentia_tool_name"
DISPLAY_NAME_TAG = "agentia_tool_display_name"
IS_TOOL_TAG = "agentia_tool_is_tool"
DESCRIPTION_TAG = "agentia_tool_description"
METADATA_TAG = "agentia_tool_metadata"


class _BaseTool:
    def __init__(
        self,
        name: str,
        schema: Any,
        display_name: str | None = None,
        description: str | None = None,
        metadata: Any | None = None,
    ):
        self.name = name
        self.display_name = display_name
        self.description = description
        self.schema = schema
        self.metadata = metadata


class _PythonFunctionTool(_BaseTool):
    def __init__(self, f: Callable[..., Any], plugin: Plugin | None = None):
        if plugin:
            name = getattr(f, NAME_TAG, None) or f"{plugin.name()}__{f.__name__}"
            display_name: str = (
                getattr(f, DISPLAY_NAME_TAG, None) or f"{plugin.name()}@{f.__name__}"
            )
        else:
            name = getattr(f, NAME_TAG, f.__name__)
            display_name = getattr(f, DISPLAY_NAME_TAG, name)
        description: str = getattr(f, DESCRIPTION_TAG, f.__doc__) or ""
        metadata = getattr(f, METADATA_TAG, None)
        params = [
            ToolFuncParam(p, self.func.__name__)
            for p in inspect.signature(f).parameters.values()
            if p.name != "self"
        ]
        schema = _PythonFunctionTool.get_json_schema(
            name=name, description=description, fparams=params
        )
        super().__init__(
            name=name,
            schema=schema,
            display_name=display_name,
            description=description,
            metadata=metadata,
        )
        self.func = f
        self.plugin = plugin
        self.params = params

    @staticmethod
    def get_json_schema(
        name: str, description: str | None, fparams: list[ToolFuncParam]
    ) -> Any:
        params: Any = {"type": "object", "properties": {}, "required": []}
        for p in fparams:
            if p.required:
                params["required"].append(p.name)
            params["properties"][p.name] = p.schema
        return {
            "name": name,
            "description": description or "",
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


class _MCPTool(_BaseTool):
    def __init__(self, name: str, schema: Any, server: MCPServer):
        print(f"Creating MCP Tool: {name}", schema)
        super().__init__(name=name, schema=schema)
        self.server = server


class ToolRegistry:
    def __init__(self, agent: "Agent", tools: Tools | None = None) -> None:
        self.__tools: dict[str, _BaseTool] = {}
        self.__plugins: list[Plugin] = []
        self.__mcp_servers: list[MCPServer] = []
        self._agent = agent
        for t in tools or []:
            self.add(t)
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

        for server in self.__mcp_servers:
            await server.init()
            tools = server.get_tools()
            for tool in tools:
                self.__tools[tool.name] = tool
            self._agent.log.info(f"Initialized MCP Server: {server.name}")

    def add(self, t: Callable[..., Any] | MCPServer | Plugin) -> None:
        if isinstance(t, Plugin):
            self.__add_plugin(t)
        elif isinstance(t, MCPServer):
            self.__add_mcp_server(t)
        else:
            assert inspect.isfunction(t), "Expected a function, MCPServer, or Plugin"
            self.__add_function(t)

    def __add_function(self, f: Callable[..., Any], plugin: Plugin | None = None):
        t = _PythonFunctionTool(f=f, plugin=plugin)
        self.__tools[t.name] = t

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

    def __add_mcp_server(self, server: MCPServer):
        self.__mcp_servers.append(server)
        self._agent.log.info(f"Added MCP Server: {server.name}")

    def get_plugin(self, type: Type[Plugin]) -> Plugin | None:
        for p in self.__plugins:
            if isinstance(p, type):
                return p
        return None

    def is_empty(self) -> bool:
        return len(self.__tools) == 0

    def get_schema(self) -> list[Any]:
        functions = [v.schema for (k, v) in self.__tools.items()]
        return [{"type": "function", "function": f} for f in functions]

    async def __run_mcp_tool(
        self,
        tool: _MCPTool,
        args: Any,
        tool_id: str | None,
    ) -> ToolResult:
        self._agent.log.info(f"TOOL#{tool_id} {tool.name} {args}")
        result = await tool.server.run(tool.name, args)
        self._agent.log.info(f"TOOL#{tool_id} {tool.name} -> {result}")
        return ToolResult(result)

    async def __run_python_tool(
        self,
        tool: _PythonFunctionTool,
        args: Any,
        tool_id: str | None,
    ) -> AsyncGenerator[Event, None]:
        func = tool.func
        args, kw_args = tool.process_json_args(self._agent, args)
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
                        yielded.tool = tool.name
                        if p := tool.plugin:
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
        result_s = json.dumps(result)
        self._agent.log.info(f"TOOL#{tool_id} {tool.name} -> {result_s}")
        raise ToolResult(result)

    async def __call_tool(
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
        if name not in self.__tools:
            raise ToolResult({"error": f"Tool `{name}` not found"})
        tool = self.__tools[name]
        if isinstance(tool, _MCPTool):
            result = await self.__run_mcp_tool(tool, args, tool_id)
            raise result
        elif isinstance(tool, _PythonFunctionTool):
            async for e in self.__run_python_tool(tool, args, tool_id):
                yield e

    async def call_tools(self, tool_calls: Sequence[ToolCall]):
        for t in tool_calls:
            assert t.type == "function"
            assert t.function.name in self.__tools
            name = t.function.name
            info = self.__tools[name]

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
                async for e in self.__call_tool(t.function, tool_id=t.id):
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


__all__ = ["Tool", "Tools", "ToolRegistry"]
