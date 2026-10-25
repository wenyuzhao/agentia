"""MCP servers as agent tools.

`MCPTools` exposes an `MCPCatalog` (any object with `catalog()` and `call()`) to the
model: one `mcp__<server>__<tool>` per tool, or `mcp_search` + `mcp_call` when there are
too many tools to list. `MCPServers` is the built-in catalog over `MCP` server objects;
hosts with their own connection management (OAuth, secrets, caching) implement the
protocol themselves.
"""

import contextlib
import inspect
import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from agentia.mcp.client import (
    MCPCallResult,
    MCPClient,
    MCPError,
    MCPSession,
    MCPToolInfo,
    StdioClient,
    Unauthorized,
    call_result,
)
from agentia.tools.tools import BaseTool, ModelTool, ToolCall, ToolProvider, ToolResult

if TYPE_CHECKING:
    from agentia.agent import Agent

_LOGGER = logging.getLogger("agentia.mcp")

SEARCH_THRESHOLD = 40

type ServerStatus = Literal["connected", "needs_auth", "error", "disabled", "idle"]


class MCPServerInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    status: ServerStatus = "connected"
    tools: int = 0
    error: str | None = None


class MCPCatalog(Protocol):
    async def catalog(self) -> tuple[Sequence[MCPServerInfo], Sequence[MCPToolInfo]]:
        """The servers and the tools of the usable ones."""
        ...

    async def call(
        self, server: str, tool: str, arguments: dict[str, Any], /
    ) -> MCPCallResult: ...


class MCP:
    """One MCP server: a local stdio command or a remote streamable-HTTP URL.

    Connections open lazily on first use and stay open until `aclose()` (the agent
    closes them in `Agent.aclose()` / `async with agent`).
    """

    def __init__(
        self,
        name: str,
        *,
        command: str | None = None,
        args: Sequence[Any] | None = None,
        env: dict[str, str] | None = None,
        cwd: Any = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        auth: str | httpx.Auth | None = None,
        timeout: float | None = None,
        client: httpx.AsyncClient | None = None,
        type: Literal["local", "stdio", "http", "streamable-http"] | None = None,
    ):
        """
        - `command`/`args`/`env`/`cwd`: a local server spoken to over stdio.
        - `url`/`headers`/`auth`: a remote server over streamable HTTP. `auth` is a
          bearer token or an `httpx.Auth`; see `agentia.mcp.oauth` for OAuth flows.
        - `timeout`: seconds to wait for one tool call (default 300).
        """
        kind = type or ("stdio" if command else "http")
        if kind in ("local", "stdio"):
            if not command:
                raise ValueError("a local MCP server needs a command")
            if url or headers or auth:
                raise ValueError("a local MCP server takes no url, headers or auth")
            self.kind: Literal["stdio", "http"] = "stdio"
        elif kind in ("http", "streamable-http"):
            if not url:
                raise ValueError("a remote MCP server needs a url")
            if command or args or env or cwd:
                raise ValueError(
                    "a remote MCP server takes no command, args, env or cwd"
                )
            self.kind = "http"
        else:
            raise ValueError(
                f"unsupported MCP transport {kind!r} (the legacy SSE transport is not "
                "supported; use streamable HTTP)"
            )
        self.name = name
        self.command = command
        self.args = [str(a) for a in args or []]
        self.env = env
        self.cwd = str(cwd) if cwd else None
        self.url = url
        self.headers = dict(headers or {})
        if isinstance(auth, str):
            self.headers.setdefault("authorization", f"Bearer {auth}")
            auth = None
        self.auth = auth
        self.timeout = timeout or 300
        self.client = client
        self.session: MCPSession | None = None
        self.tools: list[MCPToolInfo] = []

    async def connect(self) -> MCPSession:
        if self.session:
            return self.session
        session: MCPSession
        if self.kind == "stdio":
            assert self.command
            session = StdioClient(self.command, self.args, self.env, self.cwd)
        else:
            assert self.url
            auth, client = self.auth, self.client

            async def send(req: httpx.Request) -> httpx.Response:
                http = client or self._own_client()
                return await http.send(req, auth=auth or httpx.USE_CLIENT_DEFAULT)

            session = MCPClient(self.url, send, self.headers)
        await session.connect()
        self.session = session
        return session

    _own: httpx.AsyncClient | None = None

    def _own_client(self) -> httpx.AsyncClient:
        if self._own is None:
            self._own = httpx.AsyncClient()
        return self._own

    async def list_tools(self) -> list[MCPToolInfo]:
        s = await self.connect()
        self.tools = [
            MCPToolInfo.from_mcp(self.name, t) for t in await s.list_all_tools()
        ]
        return self.tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPCallResult:
        s = await self.connect()
        try:
            return call_result(await s.call_tool(name, arguments, self.timeout))
        except (MCPError, httpx.HTTPError, OSError):
            # The session may be dead (server restarted); reconnect next time.
            await self.aclose()
            raise

    async def aclose(self) -> None:
        if self.session:
            with contextlib.suppress(Exception):
                await self.session.close()
            self.session = None
        if self._own:
            await self._own.aclose()
            self._own = None


class MCPServers:
    """The built-in `MCPCatalog` over `MCP` objects. A server that fails to connect is
    reported with status `error` (or `needs_auth`) and contributes no tools."""

    def __init__(self, servers: Sequence[MCP]) -> None:
        self.servers = {s.name: s for s in servers}

    async def catalog(self) -> tuple[list[MCPServerInfo], list[MCPToolInfo]]:
        infos: list[MCPServerInfo] = []
        tools: list[MCPToolInfo] = []
        for s in self.servers.values():
            try:
                lst = s.tools if s.session and s.tools else await s.list_tools()
            except Unauthorized:
                infos.append(MCPServerInfo(name=s.name, status="needs_auth"))
                continue
            except Exception as e:  # noqa: BLE001 -- reported as the server's status
                _LOGGER.warning("MCP server %s: %s", s.name, e)
                infos.append(MCPServerInfo(name=s.name, status="error", error=str(e)))
                continue
            infos.append(MCPServerInfo(name=s.name, tools=len(lst)))
            tools += lst
        return infos, tools

    async def call(
        self, server: str, tool: str, arguments: dict[str, Any]
    ) -> MCPCallResult:
        s = self.servers.get(server)
        if not s:
            return MCPCallResult(text=f"unknown MCP server '{server}'", is_error=True)
        try:
            return await s.call_tool(tool, arguments)
        except Unauthorized:
            return MCPCallResult(text="authorization required", is_error=True)
        except (MCPError, httpx.HTTPError, OSError) as e:
            return MCPCallResult(text=f"MCP error: {e}", is_error=True)

    async def aclose(self) -> None:
        for s in self.servers.values():
            await s.aclose()


def mcp_tool_name(server: str, tool: str) -> str:
    def clean(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]", "_", s)

    return f"mcp__{clean(server)}__{clean(tool)}"[:64]


def _matches(name: str, patterns: Sequence[str]) -> bool:
    return any(re.fullmatch(re.escape(p).replace(r"\*", ".*"), name) for p in patterns)


type AuthCallback = Callable[[str, str], Awaitable[None] | None]


class MCPTool(BaseTool):
    """One MCP tool. `metadata["mcp"]` holds its server, name and `destructive` flag."""

    def __init__(self, owner: "MCPTools", info: MCPToolInfo) -> None:
        params = info.input_schema or {"type": "object", "properties": {}}
        super().__init__(
            mcp_tool_name(info.server, info.name),
            f"[MCP {info.server}] {info.description or ''}"[:1024],
            params,
            display_name=f"{info.server}/{info.name}",
            metadata={
                "mcp": {
                    "server": info.server,
                    "tool": info.name,
                    "destructive": info.destructive,
                }
            },
        )
        self.owner = owner
        self.info = info

    async def execute(self, ctx: ToolCall) -> ToolResult:
        return await self.owner.call(self.info.server, self.info.name, ctx.input)


class _SearchArgs(BaseModel):
    query: str | None = Field(None, description="Keywords; empty lists all")


class _CallArgs(BaseModel):
    server: str = Field(description="Server name")
    tool: str = Field(description="Tool name")
    arguments: dict[str, Any] | None = Field(None, description="Tool arguments")


class MCPTools(ToolProvider):
    """Expose an `MCPCatalog` as tools.

    - `mode`: `"direct"` lists every tool, `"search"` offers `mcp_search` + `mcp_call`,
      `"auto"` (default) switches to search above `search_threshold` tools.
    - `allow`: glob patterns over `mcp__<server>__<tool>`; other tools are hidden.
    - `on_auth_required(server, url)`: called when a server needs the user to sign in
      (e.g. to show them a login button); the model gets an error to retry later.
    """

    def __init__(
        self,
        catalog: MCPCatalog,
        *,
        mode: Literal["auto", "direct", "search"] = "auto",
        search_threshold: int = SEARCH_THRESHOLD,
        allow: Sequence[str] | None = None,
        on_auth_required: AuthCallback | None = None,
    ) -> None:
        self.catalog = catalog
        self.mode = mode
        self.search_threshold = search_threshold
        self.allow = allow
        self.on_auth_required = on_auth_required
        self.servers: list[MCPServerInfo] = []
        self.tools: list[MCPToolInfo] = []
        self.search_mode = False

    async def load(self) -> None:
        try:
            servers, tools = await self.catalog.catalog()
        except Exception as e:  # noqa: BLE001 -- MCP trouble must not break the agent
            _LOGGER.warning("MCP catalog failed: %s", e)
            servers, tools = [], []
        if self.allow:
            tools = [
                t
                for t in tools
                if _matches(mcp_tool_name(t.server, t.name), self.allow)
            ]
        self.servers, self.tools = list(servers), list(tools)
        self.search_mode = self.mode == "search" or (
            self.mode == "auto" and len(self.tools) > self.search_threshold
        )

    async def get_tools(self, agent: "Agent") -> list[BaseTool]:
        await self.load()
        if not self.search_mode:
            return [MCPTool(self, t) for t in self.tools]
        return [
            ModelTool(
                "mcp_search",
                "Search the available MCP tools by keyword. Returns tool names, descriptions and input schemas.",
                _SearchArgs,
                lambda _ctx, a: self.search(a.query),
                display_name="mcp_search",
            ),
            ModelTool(
                "mcp_call",
                "Call an MCP tool found with mcp_search.",
                _CallArgs,
                lambda _ctx, a: self.call(a.server, a.tool, a.arguments or {}),
                display_name="mcp_call",
            ),
        ]

    def find(self, server: str, tool: str) -> MCPToolInfo | None:
        return next(
            (t for t in self.tools if t.server == server and t.name == tool), None
        )

    def search(self, query: str | None) -> str:
        words = (query or "").lower().split()
        hits = [
            t
            for t in self.tools
            if all(
                w in f"{t.server} {t.name} {t.description or ''}".lower() for w in words
            )
        ][:15]
        if not hits:
            return "no matching MCP tools"
        return "\n\n".join(
            f"{t.server}/{t.name}: {t.description or ''}\n  input: {json.dumps(t.input_schema or {}, separators=(',', ':'))[:1500]}"
            for t in hits
        )

    async def call(
        self, server: str, tool: str, arguments: dict[str, Any]
    ) -> ToolResult:
        if self.search_mode and not self.find(server, tool):
            return ToolResult(
                output=f"unknown MCP tool {server}/{tool}; use mcp_search",
                is_error=True,
            )
        r = await self.catalog.call(server, tool, arguments)
        if r.auth_url:
            if self.on_auth_required:
                x = self.on_auth_required(server, r.auth_url)
                if inspect.isawaitable(x):
                    await x
            return ToolResult(
                output=f"MCP server '{server}' needs authorization. I've sent the user a login button; retry after they authorize.",
                is_error=True,
            )
        return ToolResult(output=r.text, is_error=r.is_error, files=r.files)

    async def get_instructions(self) -> str | None:
        if not self.servers:
            return None
        lines = [f"- {s.name}: {s.status}, {s.tools} tools" for s in self.servers]
        tail = (
            "MCP tools are not listed individually: use mcp_search to find one and mcp_call to call it."
            if self.search_mode
            else "MCP tools are named mcp__<server>__<tool>."
        )
        return "# MCP servers\n" + "\n".join(lines) + "\n" + tail

    async def aclose(self) -> None:
        close = getattr(self.catalog, "aclose", None)
        if close:
            await close()


__all__ = [
    "MCP",
    "MCPCatalog",
    "MCPServerInfo",
    "MCPServers",
    "MCPTool",
    "MCPTools",
    "mcp_tool_name",
]
