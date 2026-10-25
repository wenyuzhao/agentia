"""Tools: what the model can call.

Sources, all accepted in `Agent(tools=[...])`:
- plain functions (schema from the signature, description from the docstring),
- `Plugin`s (their `@tool` methods plus `Plugin.tools()`),
- `BaseTool` subclasses, e.g. `ModelTool` with a pydantic arguments model,
- `ToolProvider`s, whose tools are only known at run time (e.g. `MCPTools`),
- `MCP` servers (wrapped in an `MCPTools` provider),
- `ProviderTool`s, built-in tools executed by the provider.

A tool returns a `ToolResult`, any JSON value, or `Suspend` to pause the run. Tools that
share an `exclusive()` key run one at a time; the rest of a batch runs in parallel.
"""

import abc
import inspect
import json
import logging
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from inspect import Parameter
from typing import TYPE_CHECKING, Any, Optional, get_args, get_origin

from pydantic import BaseModel, Field, JsonValue, ValidationError
from pydantic.type_adapter import TypeAdapter

from agentia import models
from agentia.models.base import File
from agentia.utils.decorators import ToolFuncParam
from agentia.utils.schema import schema_of_model

NAME_TAG = "agentia_tool_name"
DISPLAY_NAME_TAG = "agentia_tool_display_name"
IS_TOOL_TAG = "agentia_tool_is_tool"
DESCRIPTION_TAG = "agentia_tool_description"
METADATA_TAG = "agentia_tool_metadata"

_LOGGER = logging.getLogger("agentia.tools")

if TYPE_CHECKING:
    from agentia.agent import Agent
    from agentia.plugins import Plugin
    from agentia.tools.mcp import MCP


class ToolResult(BaseModel):
    output: JsonValue = None
    files: list[File] = Field(default_factory=list)
    is_error: bool = False
    meta: Any = None
    """Host-defined metadata, copied onto the `FunctionCallOutput`."""


@dataclass
class Suspend:
    """Returned by a tool (or `Agent.before_tool_call`) to pause the run.

    The call stays pending (no output is recorded) and the run ends with finish reason
    `suspended`. To continue, either add the call's `FunctionCallOutput` to the history
    yourself, or let the tool run again, then call `agent.run(None)`. A tool that may
    be re-run must be idempotent by `call_id`.
    """

    reason: str | None = None


class ToolInputError(ValueError):
    """Invalid tool arguments; reported to the model as-is."""


@dataclass
class ToolCall:
    """The context of one tool invocation. Python tools get it by annotating a
    parameter with `ToolCall`."""

    agent: "Agent"
    call: models.FunctionCall
    tool: "BaseTool"

    @property
    def call_id(self) -> str:
        return self.call.call_id

    @property
    def input(self) -> dict[str, JsonValue]:
        """The parsed arguments (raises `ToolInputError` on invalid JSON)."""
        try:
            return self.call.input
        except (ValueError, TypeError) as e:
            raise ToolInputError(f"invalid JSON arguments: {e}") from e


class BaseTool(abc.ABC):
    """A tool with an explicit JSON schema."""

    def __init__(
        self,
        name: str,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        *,
        display_name: str | None = None,
        metadata: Any = None,
        strict: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters: dict[str, Any] = parameters or {
            "type": "object",
            "properties": {},
        }
        self.display_name = display_name or name
        self.metadata = metadata
        self.strict = strict

    def schema(self) -> models.FunctionTool:
        return models.FunctionTool(
            name=self.name,
            description=self.description,
            input_schema=self.parameters,
            strict=self.strict,
        )

    def exclusive(self, args: dict[str, Any]) -> str | None:
        """Calls with the same key never run concurrently (e.g. writes to one file)."""
        return None

    @abc.abstractmethod
    async def execute(self, ctx: ToolCall) -> "ToolResult | Suspend | JsonValue": ...


class ModelTool[A: BaseModel](BaseTool):
    """A tool whose arguments are a pydantic model. Override `run`, or pass `run=`."""

    def __init__(
        self,
        name: str,
        description: str,
        args: type[A],
        run: "Callable[[ToolCall, A], Any] | None" = None,
        *,
        parameters: dict[str, Any] | None = None,
        exclusive: Callable[[A], str | None] | None = None,
        display_name: str | None = None,
        metadata: Any = None,
    ) -> None:
        super().__init__(
            name,
            description,
            parameters if parameters is not None else schema_of_model(args),
            display_name=display_name,
            metadata=metadata,
        )
        self.args = args
        self._run = run
        self._exclusive = exclusive

    def parse(self, raw: dict[str, Any]) -> A:
        try:
            return self.args.model_validate(raw)
        except ValidationError as e:
            raise ToolInputError(f"invalid arguments: {validation_message(e)}") from e

    def exclusive(self, args: dict[str, Any]) -> str | None:
        if not self._exclusive:
            return None
        try:
            return self._exclusive(self.args.model_validate(args))
        except ValidationError:
            return None

    async def run(self, ctx: ToolCall, args: A) -> "ToolResult | Suspend | JsonValue":
        if self._run is None:
            raise NotImplementedError(f"tool {self.name} has no run function")
        r = self._run(ctx, args)
        return await r if inspect.isawaitable(r) else r

    async def execute(self, ctx: ToolCall) -> "ToolResult | Suspend | JsonValue":
        return await self.run(ctx, self.parse(ctx.input))


def validation_message(e: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(x) for x in err['loc']) or 'arguments'}: {err['msg']}"
        for err in e.errors()
    )


def _injected(annotation: Any) -> str | None:
    """'agent' / 'call' for parameters filled by agentia rather than the model."""
    from agentia.agent import Agent

    if get_origin(annotation) is not None and get_args(annotation):
        annotation = get_args(annotation)[0]
    if inspect.isclass(annotation):
        if issubclass(annotation, Agent):
            return "agent"
        if issubclass(annotation, ToolCall):
            return "call"
    return None


class PythonTool(BaseTool):
    """A tool from a Python function or a plugin method."""

    def __init__(self, f: Callable[..., Any], plugin: Optional["Plugin"] = None):
        if plugin:
            name = getattr(f, NAME_TAG, None) or f"{plugin.name()}__{f.__name__}"
            display_name: str = (
                getattr(f, DISPLAY_NAME_TAG, None) or f"{plugin.name()}@{f.__name__}"
            )
        else:
            name = getattr(f, NAME_TAG, f.__name__)
            display_name = getattr(f, DISPLAY_NAME_TAG, name)
        description: str = textwrap.dedent(
            getattr(f, DESCRIPTION_TAG, f.__doc__) or ""
        ).strip()
        sig = inspect.signature(f)
        self.injected = {
            p.name: kind
            for p in sig.parameters.values()
            if (kind := _injected(p.annotation))
        }
        self.params = [
            ToolFuncParam(p, f.__name__)
            for p in sig.parameters.values()
            if p.name != "self" and p.name not in self.injected
        ]
        schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        for p in self.params:
            if p.required:
                schema["required"].append(p.name)
            schema["properties"][p.name] = p.schema
        super().__init__(
            name,
            description,
            schema,
            display_name=display_name,
            metadata=getattr(f, METADATA_TAG, None),
        )
        self.func = f
        self.plugin = plugin

    def _arguments(
        self, ctx: ToolCall, args: dict[str, Any]
    ) -> tuple[list[Any], dict[str, Any]]:
        p_args: list[Any] = []
        kw_args: dict[str, Any] = {}
        sig = inspect.signature(self.func)
        by_name = {p.name: p for p in self.params}
        for param in sig.parameters.values():
            if param.name == "self":
                continue
            if kind := self.injected.get(param.name):
                value: Any = ctx.agent if kind == "agent" else ctx
            else:
                p = by_name[param.name]
                if param.name not in args:
                    value = p.default
                else:
                    try:
                        value = TypeAdapter(p.type).validate_python(args[param.name])
                    except ValidationError as e:
                        raise ToolInputError(
                            f"Error parsing argument '{param.name}': {e}. Please retry with a valid value."
                        ) from e
            match param.kind:
                case Parameter.POSITIONAL_ONLY:
                    p_args.append(value)
                case Parameter.POSITIONAL_OR_KEYWORD | Parameter.KEYWORD_ONLY:
                    kw_args[param.name] = value
                case other:
                    raise ValueError(f"{other} is not supported")
        return p_args, kw_args

    async def execute(self, ctx: ToolCall) -> "ToolResult | Suspend | JsonValue":
        p_args, kw_args = self._arguments(ctx, ctx.input)
        output = self.func(*p_args, **kw_args)
        if inspect.isawaitable(output):
            output = await output
        return output


class ProviderTool(BaseModel):
    """A built-in tool executed by the provider (e.g. `web_search`)."""

    name: str
    args: dict[str, JsonValue] | None = None


class ToolProvider(abc.ABC):
    """A source of tools that are only known at run time (e.g. an MCP catalog)."""

    @abc.abstractmethod
    async def get_tools(self, agent: "Agent") -> Sequence[BaseTool]: ...

    async def get_instructions(self) -> str | None:
        return None

    async def aclose(self) -> None:
        """Release connections. Called when the agent is closed."""


type Tool = Plugin | Callable[..., Any] | ProviderTool | MCP | BaseTool | ToolProvider
type Tools = Sequence[Tool]


def to_output(
    call: models.FunctionCall, result: "ToolResult | JsonValue"
) -> models.FunctionCallOutput:
    if isinstance(result, ToolResult):
        return models.FunctionCallOutput.create(
            call_id=call.call_id,
            name=call.name,
            output=result.output,
            files=result.files,
            is_error=result.is_error or None,
            meta=result.meta,
        )
    return models.FunctionCallOutput.create(
        call_id=call.call_id, name=call.name, output=result
    )


def error_output(call: models.FunctionCall, message: str) -> models.FunctionCallOutput:
    return models.FunctionCallOutput.create(
        call_id=call.call_id, name=call.name, output=message, is_error=True
    )


class ToolSet:
    def __init__(self, tools: Tools, agent: "Agent"):
        from agentia.plugins import Plugin
        from agentia.tools.mcp import MCP, MCPServers, MCPTools

        self.agent = agent
        self.plugins: dict[str, Plugin] = {}
        self.providers: list[ToolProvider] = []
        self.provider_tools: dict[str, ProviderTool] = {}
        self.__static: dict[str, BaseTool] = {}
        self.__dynamic: dict[str, BaseTool] = {}
        servers: list[MCP] = []
        for t in tools:
            if isinstance(t, Plugin):
                self.__add_plugin(t)
            elif isinstance(t, MCP):
                servers.append(t)
            elif isinstance(t, ProviderTool):
                self.provider_tools[t.name] = t
            elif isinstance(t, BaseTool):
                self.__static[t.name] = t
            elif isinstance(t, ToolProvider):
                self.providers.append(t)
            else:
                assert inspect.isfunction(t) or inspect.ismethod(t), (
                    f"Unsupported tool: {t!r}"
                )
                self.__add_function(t)
        if servers:
            self.mcp_servers = MCPServers(servers)
            self.providers.append(MCPTools(self.mcp_servers))
        self.__initialized = False

    async def init(self) -> None:
        """Initialize plugins once, and reload dynamic tools (every run)."""
        if not self.__initialized:
            self.__initialized = True
            for plugin in self.plugins.values():
                try:
                    plugin._agent = self.agent
                    await plugin.init()
                except Exception as e:
                    from . import PluginInitError

                    raise PluginInitError(plugin.id(), e) from e
        dynamic: dict[str, BaseTool] = {}
        for p in self.providers:
            for t in await p.get_tools(self.agent):
                dynamic[t.name] = t
        self.__dynamic = dynamic

    async def aclose(self) -> None:
        for p in self.providers:
            await p.aclose()

    def add(self, t: "Callable[..., Any] | Plugin | BaseTool | ToolProvider") -> None:
        from agentia.plugins import Plugin

        if isinstance(t, Plugin):
            self.__add_plugin(t)
        elif isinstance(t, BaseTool):
            self.__static[t.name] = t
        elif isinstance(t, ToolProvider):
            self.providers.append(t)
        else:
            assert inspect.isfunction(t) or inspect.ismethod(t), (
                "Expected a function, BaseTool, ToolProvider or Plugin"
            )
            self.__add_function(t)

    def __add_function(self, f: Callable[..., Any], plugin: Optional["Plugin"] = None):
        t = PythonTool(f=f, plugin=plugin)
        self.__static[t.name] = t

    def __add_plugin(self, p: "Plugin"):
        for _n, method in inspect.getmembers(p, predicate=inspect.ismethod):
            if not getattr(method, IS_TOOL_TAG, False):
                continue
            self.__add_function(method, plugin=p)
        for t in p.tools():
            self.__static[t.name] = t
        self.plugins[p.id()] = p

    def get_plugin[T: "Plugin"](self, type: type[T]) -> T | None:
        for p in self.plugins.values():
            if isinstance(p, type):
                return p
        return None

    def get(self, name: str) -> BaseTool | None:
        return self.__dynamic.get(name) or self.__static.get(name)

    def all(self) -> list[BaseTool]:
        return list({**self.__static, **self.__dynamic}.values())

    async def get_instructions(self) -> list[str]:
        out: list[str] = []
        sources: list[Any] = [*self.plugins.values(), *self.providers]
        for s in sources:
            ins = s.get_instructions()
            if inspect.isawaitable(ins):
                ins = await ins
            if ins and (t := ins.strip()):
                out.append(t)
        return out

    def is_empty(self) -> bool:
        return not self.all() and not self.provider_tools

    def get_schema(self) -> Sequence[models.ToolSchema]:
        functions: list[models.ToolSchema] = [t.schema() for t in self.all()]
        provider_tools: list[models.ToolSchema] = [
            models.ProviderDefinedTool(id=t.name, name=t.name, args=t.args or {})
            for t in self.provider_tools.values()
        ]
        return functions + provider_tools

    def exclusive_key(self, call: models.FunctionCall) -> str | None:
        tool = self.get(call.name)
        if not tool:
            return None
        try:
            return tool.exclusive(call.input)
        except (ValueError, TypeError):
            return None

    async def execute(
        self, call: models.FunctionCall
    ) -> "models.FunctionCallOutput | Suspend":
        """Run one call. Failures become error outputs; they never abort the run."""
        tool = self.get(call.name)
        if tool is None:
            return error_output(call, f"unknown tool '{call.name}'")
        try:
            result = await tool.execute(
                ToolCall(agent=self.agent, call=call, tool=tool)
            )
        except ToolInputError as e:
            return error_output(call, str(e))
        except Exception as e:
            _LOGGER.exception("tool %s failed", call.name)
            return error_output(call, json.dumps({"error": str(e) or type(e).__name__}))
        if isinstance(result, Suspend):
            return result
        return to_output(call, result)


__all__ = [
    "BaseTool",
    "ModelTool",
    "ProviderTool",
    "PythonTool",
    "Suspend",
    "Tool",
    "ToolCall",
    "ToolInputError",
    "ToolProvider",
    "ToolResult",
    "ToolSet",
    "Tools",
    "error_output",
    "to_output",
    "validation_message",
]
