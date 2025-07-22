from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Callable
import weakref
from git import TYPE_CHECKING
from mcp import ClientSession, StdioServerParameters, stdio_client
from pydantic import BaseModel, Field
import asyncio


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


class MCPServer:
    def __init__(
        self, name: str, cmd: list[str | Path], envs: dict[str, str] | None = None
    ):
        self.name = name
        self.config = MCPServerConfig(
            command=str(cmd[0]),
            args=[str(arg) for arg in cmd[1:]],
            env=envs or {},
        )
        self.initialized = False
        self.context_available = False

    @staticmethod
    def __convert_tool_format(tool):
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

    def get_tools(self) -> list["_MCPTool"]:
        return self.__tools

    async def init(self):
        from agentia.tools import _MCPTool

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
        await self.session.initialize()
        self.__tools: list[_MCPTool] = []
        tools = await self.session.list_tools()
        print("MCP Server is starting with tools:", tools.tools)
        for tool in tools.tools:
            schema = self.__convert_tool_format(tool)
            self.__tools.append(_MCPTool(name=tool.name, schema=schema, server=self))
        while tools.nextCursor:
            tools = await self.session.list_tools(cursor=tools.nextCursor)
            for tool in tools.tools:
                schema = self.__convert_tool_format(tool)
                self.__tools.append(
                    _MCPTool(name=tool.name, schema=schema, server=self)
                )
        print("MCP Server started with tools:", self.__tools)
        self.initialized = True

    async def run(self, tool: str, args: Any) -> Any:
        assert self.context_available, "Invalid or closed MCP context"
        result = await self.session.call_tool(tool, args)
        assert self.context_available, "Invalid or closed MCP context"
        result_json = []
        for part in result.content:
            result_json.append(part.model_dump())
        print(f"Tool {tool} called with args {args}, result: {result_json}")
        return result_json
