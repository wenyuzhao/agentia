from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Literal, Optional, overload
from typing import TYPE_CHECKING, Sequence
from fastmcp import Client
from fastmcp.mcp_config import (
    MCPConfig,
    CanonicalMCPServerTypes,
    StdioMCPServer,
    RemoteMCPServer,
)
from httpx import Auth
from mcp.types import Tool as FastMCPTool


if TYPE_CHECKING:  # pragma: no cover
    from agentia.tools.tools import _MCPTool
    from agentia.agent import Agent, LLM


class MCPContext:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self._servers: list[MCP] = []
        self.active = False

    async def __aenter__(self):
        self.active = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.active = False
        for server in self._servers:
            server.context_available = False
        await self.exit_stack.aclose()


class _ClientWrapper(Client):
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await super().__aexit__(exc_type, exc_val, exc_tb)
        await self.transport.close()


def _convert_tool_format(tool: FastMCPTool) -> Any:
    converted_tool = {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.inputSchema,
    }
    return converted_tool


class MCP:
    @overload
    def __init__(
        self,
        name: str,
        *,
        command: str,
        args: Sequence[str | Path],
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        timeout: int | None = None,
        type: Literal["local"] = "local",
        context: Optional[MCPContext] = None,
    ): ...

    @overload
    def __init__(
        self,
        name: str,
        *,
        type: Literal["http", "sse", "streamable-http"],
        url: str,
        headers: dict[str, str] | None = None,
        auth: str | Literal["oauth"] | Auth | None = None,
        timeout: int | None = None,
        context: Optional[MCPContext] = None,
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
        auth: str | Literal["oauth"] | Auth | None = None,
        timeout: int | None = None,
        type: Literal["sse", "http", "streamable-http", "local"] = "local",
        context: Optional[MCPContext] = None,
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
        self.session: Client | None = None
        self.timeout: int | None = timeout  # Maximum response time in milliseconds

        self.context = context

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
            case "http" | "sse" | "streamable-http":
                assert_no_local_args()
                assert url is not None, "URL must be specified for HTTP server"

    def get_tools(self) -> list["_MCPTool"]:
        assert self.initialized, "MCP Server not initialized"
        return self.__tools

    async def init(self, llm: "LLM", agent: Optional["Agent"] = None) -> None:
        from agentia.tools.tools import _MCPTool

        assert not self.initialized, "MCP Server already initialized"

        context: MCPContext
        if self.context:
            context = self.context
        elif agent and agent._mcp_context and agent._mcp_context.active:
            context = agent._mcp_context
        elif llm and llm._mcp_context and llm._mcp_context.active:
            context = llm._mcp_context
        else:
            raise RuntimeError("MCP Server must be running in an MCP context")

        exit_stack = context.exit_stack
        self.context_available = True
        context._servers.append(self)

        config: CanonicalMCPServerTypes
        match self.type:
            case "local":
                assert self.cmd is not None
                config = StdioMCPServer(
                    command=str(self.cmd[0]),
                    args=[str(a) for a in self.cmd[1:]],
                    env=self.env or {},
                    cwd=str(self.cwd) if self.cwd else None,
                    timeout=self.timeout,
                )
            case "sse" | "streamable-http" | "http":
                assert self.url is not None
                config = RemoteMCPServer(
                    url=self.url,
                    transport=self.type,
                    headers=self.headers or {},
                    auth=self.auth,
                    timeout=self.timeout,
                )
            case _:
                raise ValueError(f"Unsupported MCP server type: {self.type}")

        self.session = await exit_stack.enter_async_context(
            _ClientWrapper(MCPConfig(mcpServers={self.name: config}))
        )

        tools = await self.session.list_tools()
        for tool in tools:
            schema = _convert_tool_format(tool)
            self.__tools.append(_MCPTool(name=tool.name, schema=schema, server=self))
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
