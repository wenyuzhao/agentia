from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Callable, Literal, overload
from typing import TYPE_CHECKING, Sequence
from mcp import ClientSession, StdioServerParameters, stdio_client, Tool as MCPTool
from mcp.client.websocket import websocket_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client
from pydantic import BaseModel
import asyncio
from httpx import Auth


if TYPE_CHECKING:  # pragma: no cover
    from agentia.tools import _MCPTool


class MCPServerConfig(BaseModel):
    command: str
    args: list[str]
    env: dict[str, str] | None = None


class MCPContext:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.servers: list[MCPServer] = []

    async def __aenter__(self):
        global MCP_CONTEXTS
        MCP_CONTEXTS.append(self)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        global MCP_CONTEXTS
        MCP_CONTEXTS.remove(self)
        for server in self.servers:
            server.context_available = False
        await self.exit_stack.aclose()


MCP_CONTEXTS: list[MCPContext] = []


def mcp_context(f: Callable[..., Any]) -> Callable[..., Any]:
    """
    Context manager for MCP server initialization.
    This ensures that the MCP server is initialized and cleaned up properly.
    """
    assert asyncio.iscoroutinefunction(f), "Function must be async"

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        async with MCPContext() as ctx:
            return await f(*args, **kwargs)

    return wrapper


def _convert_tool_format(tool: MCPTool) -> Any:
    converted_tool = {
        "name": tool.name,
        "description": tool.description,
        "parameters": {
            "type": "object",
            "properties": tool.inputSchema["properties"],
            "required": tool.inputSchema.get("required", []),
        },
    }
    return converted_tool


class MCPServer:
    @overload
    def __init__(
        self,
        name: str,
        *,
        command: str,
        args: Sequence[str | Path],
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        type: Literal["local"] = "local",
    ): ...

    @overload
    def __init__(
        self,
        name: str,
        *,
        type: Literal["http", "sse"],
        url: str,
        headers: dict[str, str] | None = None,
        auth: Auth | None = None,
    ): ...

    @overload
    def __init__(
        self,
        name: str,
        *,
        type: Literal["websocket"],
        url: str,
    ): ...

    def __init__(
        self,
        name: str,
        *,
        command: str | None = None,
        args: Sequence[str | Path] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        auth: Auth | None = None,
        type: Literal["websocket", "sse", "http", "local"] = "local",
    ):

        self.cmd = None
        if command:
            self.cmd = [command] + ([a for a in args] if args else [])
        self.env = env
        self.cwd = cwd
        self.url = url
        self.headers = headers
        self.auth = auth
        self.type = type

        self.name = name
        self.initialized = False
        self.context_available = False
        self.__tools: list["_MCPTool"] = []
        self.session: ClientSession | None = None

        def assert_no_local_args():
            assert command is None, "Remote MCP server does not support command"
            assert args is None, "Remote MCP server does not support args"
            assert env is None, "Remote MCP server does not support env"
            assert cwd is None, "Remote MCP server does not support cwd"

        match type:
            case "local":
                assert url is None, "Local MCP server does not support URL"
                assert headers is None, "Local MCP server does not support headers"
                assert auth is None, "Local MCP server does not support auth"
                assert (
                    command is not None and len(command) > 0
                ), "Command must be specified"
                assert args is not None, "Args must be specified"
            case "http" | "sse":
                assert_no_local_args()
                assert url is not None, "URL must be specified for HTTP server"
            case "websocket":
                assert_no_local_args()
                assert url is not None, "URL must be specified for WebSocket server"
                assert headers is None, "WebSocket does not support headers"
                assert auth is None, "WebSocket does not support auth"

    def get_tools(self) -> list["_MCPTool"]:
        assert self.initialized, "MCP Server not initialized"
        return self.__tools

    async def init(self):
        from agentia.tools import _MCPTool

        assert not self.initialized, "MCP Server already initialized"
        assert len(MCP_CONTEXTS) > 0, "Agents must be running in an MCP context"

        exit_stack = MCP_CONTEXTS[-1].exit_stack
        self.context_available = True
        MCP_CONTEXTS[-1].servers.append(self)

        match self.type:
            case "local":
                assert self.cmd is not None
                server_params = StdioServerParameters(
                    command=str(self.cmd[0]),
                    args=[str(a) for a in self.cmd[1:]],
                    env=self.env,
                    cwd=self.cwd,
                )
                self.stdio, self.write = await exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
            case "websocket":
                assert self.url is not None
                self.stdio, self.write = await exit_stack.enter_async_context(
                    websocket_client(self.url)
                )
            case "sse":
                assert self.url is not None
                self.stdio, self.write = await exit_stack.enter_async_context(
                    sse_client(
                        self.url,
                        headers=self.headers,
                        auth=self.auth,
                    )
                )
            case "http":
                assert self.url is not None
                self.stdio, self.write, _ = await exit_stack.enter_async_context(
                    streamablehttp_client(
                        self.url, headers=self.headers, auth=self.auth
                    )
                )
            case _:
                raise ValueError(f"Unsupported MCP server type: {self.type}")
        self.session = await exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()
        tools = await self.session.list_tools()
        for tool in tools.tools:
            schema = _convert_tool_format(tool)
            self.__tools.append(_MCPTool(name=tool.name, schema=schema, server=self))
        while tools.nextCursor:
            tools = await self.session.list_tools(cursor=tools.nextCursor)
            for tool in tools.tools:
                schema = _convert_tool_format(tool)
                self.__tools.append(
                    _MCPTool(name=tool.name, schema=schema, server=self)
                )
        self.initialized = True

    async def run(self, tool: str, args: Any) -> Any:
        assert self.context_available, "Invalid or closed MCP context"
        assert self.session is not None, "MCP session not initialized"
        result = await self.session.call_tool(tool, args)
        assert self.context_available, "Invalid or closed MCP context"
        result_json = []
        for part in result.content:
            result_json.append(part.model_dump())
        return result_json
