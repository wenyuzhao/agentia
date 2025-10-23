import inspect
import json
import os
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union

from pydantic import AliasChoices, BaseModel, Field, JsonValue
from agentia.tools.mcp import MCPServer
import agentia.spec as spec
from agentia.utils.decorators import ToolFuncParam
from inspect import Parameter
from uuid import uuid4

NAME_TAG = "agentia_tool_name"
DISPLAY_NAME_TAG = "agentia_tool_display_name"
IS_TOOL_TAG = "agentia_tool_is_tool"
DESCRIPTION_TAG = "agentia_tool_description"
METADATA_TAG = "agentia_tool_metadata"

if TYPE_CHECKING:
    from ..llm import LLM
    from agentia.plugins import Plugin
    from agentia.agent import Agent


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
    def __init__(self, f: Callable[..., Any], plugin: Optional["Plugin"] = None):
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
            ToolFuncParam(p, f.__name__)
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

    def process_json_args(self, llm: "LLM", args: Any):
        from agentia.agent import Agent
        from ..llm import LLM

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
            is_model = inspect.isclass(p.type) and issubclass(p.type, BaseModel)
            match p.param.kind:
                case Parameter.POSITIONAL_ONLY if p.param.annotation == Agent:
                    if llm._agent is not None:
                        p_args.append(llm._agent)
                case Parameter.POSITIONAL_ONLY if p.param.annotation == LLM:
                    p_args.append(llm)
                case Parameter.POSITIONAL_ONLY:
                    default = p.default if p.default != Parameter.empty else None
                    p_args.append(get_value(p.name, default, is_model))
                case Parameter.POSITIONAL_OR_KEYWORD | Parameter.KEYWORD_ONLY if (
                    p.param.annotation == Agent
                ):
                    if llm._agent is not None:
                        kw_args[p.name] = llm._agent
                case Parameter.POSITIONAL_OR_KEYWORD | Parameter.KEYWORD_ONLY:
                    default = p.default if p.default != Parameter.empty else None
                    kw_args[p.name] = get_value(p.name, default, is_model)
                case other:
                    raise ValueError(f"{other} is not supported")
        return p_args, kw_args


class _MCPTool(_BaseTool):
    def __init__(self, name: str, schema: Any, server: MCPServer):
        super().__init__(name=name, schema=schema)
        self.server = server


class ProviderTool(BaseModel):
    name: str
    args: dict[str, JsonValue] | None = None


type Tool = Plugin | Callable[..., Any] | ProviderTool | MCPServer
type Tools = Sequence[Tool]


class FileResult(BaseModel):
    id: str | None = None
    data: spec.DataContent
    media_type: str = Field(
        validation_alias=AliasChoices("mediaType", "media_type"),
        serialization_alias="mediaType",
    )


class ToolSet:
    def __init__(self, tools: Tools):
        from agentia.plugins import Plugin

        self.all_tools = tools
        self.plugins: dict[str, Plugin] = {}
        self.mcp_servers: dict[str, MCPServer] = {}
        self.provider_tools: dict[str, ProviderTool] = {}
        self.__tools: dict[str, _BaseTool] = {}
        for t in tools:
            if isinstance(t, Plugin):
                self.__add_plugin(t)
            elif isinstance(t, MCPServer):
                self.__add_mcp_server(t)
            elif isinstance(t, ProviderTool):
                self.provider_tools[t.name] = t
            else:
                assert inspect.isfunction(
                    t
                ), "Expected a function, MCPServer, Plugin, or ProviderTool"
                self.__add_function(t)
        self.__initialized = False

    async def init(self, llm: "LLM", agent: Optional["Agent"] = None):
        if self.__initialized:
            return
        self.__initialized = True
        for plugin in self.plugins.values():
            try:
                plugin.llm = llm
                plugin.agent = agent
                await plugin.init()
            except Exception as e:
                from . import PluginInitError

                raise PluginInitError(plugin.id(), e) from e
        for server in self.mcp_servers.values():
            await server.init()
            tools = server.get_tools()
            for tool in tools:
                self.__tools[tool.name] = tool

    def add(self, t: Union[Callable[..., Any], MCPServer, "Plugin"]) -> None:
        from agentia.plugins import Plugin

        if isinstance(t, Plugin):
            self.__add_plugin(t)
        elif isinstance(t, MCPServer):
            self.__add_mcp_server(t)
        else:
            assert inspect.isfunction(t), "Expected a function, MCPServer, or Plugin"
            self.__add_function(t)

    def __add_function(self, f: Callable[..., Any], plugin: Optional["Plugin"] = None):
        t = _PythonFunctionTool(f=f, plugin=plugin)
        self.__tools[t.name] = t

    def __add_plugin(self, p: "Plugin"):
        # Add all functions from the plugin
        for _n, method in inspect.getmembers(p, predicate=inspect.ismethod):
            if not getattr(method, IS_TOOL_TAG, False):
                continue
            self.__add_function(method, plugin=p)
        # Add the plugin to the list of plugins
        self.plugins[p.id()] = p

    def __add_mcp_server(self, server: MCPServer):
        self.mcp_servers[server.name] = server

    def get_plugin(self, type: type["Plugin"]) -> Optional["Plugin"]:
        for p in self.plugins.values():
            if isinstance(p, type):
                return p
        return None

    def is_empty(self) -> bool:
        return len(self.__tools) == 0

    def get_schema(self) -> Sequence[spec.ProviderDefinedTool | spec.FunctionTool]:
        functions = [
            spec.FunctionTool(
                name=v.name,
                description=v.description or "",
                input_schema=v.schema["parameters"],
            )
            for (k, v) in self.__tools.items()
        ]
        provider_tools = [
            spec.ProviderDefinedTool(id=t.name, name=t.name, args=t.args or {})
            for t in self.provider_tools.values()
        ]
        return functions + provider_tools

    async def __run_python_tool(
        self,
        id: str,
        tool: _PythonFunctionTool,
        llm: "LLM",
        args: Any,
    ) -> tuple[spec.ToolMessage, spec.ToolResult, FileResult | None]:
        p_args, kw_args = tool.process_json_args(llm, args)
        output = tool.func(*p_args, **kw_args)  # type: ignore
        if inspect.isawaitable(output):
            output = await output
        # if output is a FileResult, transform the result
        file = None
        tool_vision = os.getenv("AGENTIA_EXPERIMENTAL_TOOL_FILES", None)
        if tool_vision and tool_vision != "0" and tool_vision.lower() != "false":
            if isinstance(output, FileResult):
                file = output
                if not file.id:
                    file.id = str(uuid4())
                output: Any = {
                    "file_id": file.id,
                    "media_type": file.media_type,
                    "hint": "The tool outputed a file with the given file_id. The file is attached below.",
                }
                d = file.data
                if isinstance(d, str) and (d.startswith(("http://", "https://"))):
                    output["url"] = d
        else:
            if isinstance(output, FileResult):
                output = output.model_dump()
                if isinstance(output["data"], str) and output["data"].startswith(
                    "data:"
                ):
                    del output["data"]
                output["hint"] = "The tool outputed a file with the given file_id."

        tm = spec.ToolMessage(
            content=[
                spec.MessagePartToolResult(
                    tool_call_id=id,
                    tool_name=tool.name,
                    output=spec.ToolResultOutputJson(value=output),
                )
            ]
        )
        tr = spec.ToolResult(
            tool_call_id=id,
            tool_name=tool.name,
            result=output,
        )
        return tm, tr, file

    async def __run_mcp_tool(
        self, id: str, tool: _MCPTool, args: Any
    ) -> tuple[spec.ToolMessage, spec.ToolResult]:
        result = await tool.server.run(tool.name, args)
        tm = spec.ToolMessage(
            content=[
                spec.MessagePartToolResult(
                    tool_call_id=id,
                    tool_name=tool.name,
                    output=spec.ToolResultOutputJson(value=result),
                )
            ]
        )
        tr = spec.ToolResult(
            tool_call_id=id,
            tool_name=tool.name,
            result=result,
        )
        return tm, tr

    async def run(
        self, llm: "LLM", tool_calls: list[spec.ToolCall]
    ) -> tuple[spec.ToolMessage, list[spec.ToolResult], list[FileResult]]:
        tool_results: list[spec.MessagePartToolResult] = []
        tool_results2: list[spec.ToolResult] = []
        file_results: list[FileResult] = []
        for c in tool_calls:
            tool = self.__tools.get(c.tool_name, None)
            if not tool:
                raise ValueError(f"Tool {c.tool_name} not found")
            try:
                args = json.loads(c.input)
                if isinstance(tool, _PythonFunctionTool):
                    tm, tr, file = await self.__run_python_tool(
                        id=c.tool_call_id,
                        tool=tool,
                        llm=llm,
                        args=args,
                    )
                    tool_results.extend(tm.content)
                    tool_results2.append(tr)
                    if file:
                        file_results.append(file)

                elif isinstance(tool, _MCPTool):
                    tm, tr = await self.__run_mcp_tool(
                        id=c.tool_call_id,
                        tool=tool,
                        args=args,
                    )
                    tool_results.extend(tm.content)
                    tool_results2.append(tr)
                else:
                    raise ValueError(f"Tool {c.tool_name} is of unknown type")
            except Exception as e:
                output: JsonValue = {"error": str(e)}
                tool_results.append(
                    spec.MessagePartToolResult(
                        tool_call_id=c.tool_call_id,
                        tool_name=c.tool_name,
                        output=spec.ToolResultOutputErrorJson(value=output),
                    )
                )
        return spec.ToolMessage(content=tool_results), tool_results2, file_results
