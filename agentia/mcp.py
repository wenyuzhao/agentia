from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Callable, Literal, override
from git import TYPE_CHECKING
from mcp import ClientSession, StdioServerParameters, stdio_client, Tool as MCPTool
from mcp.client.websocket import websocket_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client
from pydantic import BaseModel
import asyncio
import abc
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
            "required": tool.inputSchema["required"],
        },
    }
    return converted_tool


class MCPServer(abc.ABC):
    def __init__(self, name: str):
        self.name = name
        self.initialized = False
        self.context_available = False
        self.__tools: list["_MCPTool"] = []
        self.session: ClientSession | None = None

    def get_tools(self) -> list["_MCPTool"]:
        assert self.initialized, "MCP Server not initialized"
        return self.__tools

    @abc.abstractmethod
    async def init(self) -> None:
        """Initialize the MCP server."""
        raise NotImplementedError()

    async def _init_session(self) -> None:
        from agentia.tools import _MCPTool

        assert self.session is not None, "MCP session not initialized"
        await self.session.initialize()
        tools = await self.session.list_tools()
        print("MCP Server is starting with tools:", tools.tools)
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
        print("MCP Server started with tools:", self.__tools)
        self.initialized = True

    async def run(self, tool: str, args: Any) -> Any:
        assert self.context_available, "Invalid or closed MCP context"
        assert self.session is not None, "MCP session not initialized"
        result = await self.session.call_tool(tool, args)
        assert self.context_available, "Invalid or closed MCP context"
        result_json = []
        for part in result.content:
            result_json.append(part.model_dump())
        print(f"Tool {tool} called with args {args}, result: {result_json}")
        return result_json


class LocalMCPServer(MCPServer):
    def __init__(
        self, name: str, cmd: list[str | Path], envs: dict[str, str] | None = None
    ):
        super().__init__(name)
        self.config = MCPServerConfig(
            command=str(cmd[0]), args=[str(arg) for arg in cmd[1:]], env=envs or {}
        )

    @override
    async def init(self):
        assert not self.initialized, "MCP Server already initialized"
        assert len(MCP_CONTEXTS) > 0, "Agents must be running in an MCP context"

        exit_stack = MCP_CONTEXTS[-1].exit_stack
        self.context_available = True
        MCP_CONTEXTS[-1].servers.append(self)

        server_params = StdioServerParameters(**self.config.model_dump())
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self._init_session()


class RemoteMCPServer(MCPServer):
    def __init__(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        auth: Auth | None = None,
        type: Literal["websocket", "sse", "http"] = "http",
    ):
        super().__init__(name)
        self.url = url
        self.headers = headers
        self.auth = auth
        self.type = type
        if type == "websocket":
            assert headers is None, "WebSocket does not support headers"
            assert auth is None, "WebSocket does not support auth"

    @override
    async def init(self):
        assert not self.initialized, "MCP Server already initialized"
        assert len(MCP_CONTEXTS) > 0, "Agents must be running in an MCP context"

        exit_stack = MCP_CONTEXTS[-1].exit_stack
        self.context_available = True
        MCP_CONTEXTS[-1].servers.append(self)

        match self.type:
            case "websocket":
                self.stdio, self.write = await exit_stack.enter_async_context(
                    websocket_client(self.url)
                )
            case "sse":
                self.stdio, self.write = await exit_stack.enter_async_context(
                    sse_client(
                        self.url,
                        headers=self.headers,
                        auth=self.auth,
                    )
                )
            case "http":
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
        await self._init_session()
