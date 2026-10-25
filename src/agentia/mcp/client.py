"""Minimal MCP clients: streamable HTTP (JSON-RPC 2.0 over POST, JSON or SSE responses)
and stdio (newline-delimited JSON-RPC with a subprocess)."""

import abc
import asyncio
import base64
import contextlib
import itertools
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from agentia.models import File

PROTOCOL_VERSION = "2025-06-18"

type Send = Callable[[httpx.Request], Awaitable[httpx.Response]]


class MCPError(Exception):
    pass


class Unauthorized(MCPError):
    """The server answered 401; `www_authenticate` points at its auth metadata."""

    def __init__(self, www_authenticate: str | None) -> None:
        super().__init__("unauthorized")
        self.www_authenticate = www_authenticate


class MCPToolInfo(BaseModel):
    """One tool of an MCP server's catalog."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    server: str
    name: str
    description: str | None = None
    input_schema: dict[str, Any] | None = Field(default=None, alias="inputSchema")
    destructive: bool = False

    @classmethod
    def from_mcp(cls, server: str, t: Mapping[str, Any]) -> "MCPToolInfo":
        schema = t.get("inputSchema")
        return cls.model_validate(
            {
                "server": server,
                "name": t["name"],
                "description": t.get("description"),
                "inputSchema": schema if isinstance(schema, dict) else None,
                "destructive": is_destructive(t),
            }
        )


def is_destructive(t: Mapping[str, Any]) -> bool:
    """From the tool's annotations, or its name when it doesn't say it's read-only."""
    ann = t.get("annotations") or {}
    return ann.get("destructiveHint") is True or (
        ann.get("readOnlyHint") is not True
        and bool(re.search(r"(?i)delete|remove|drop|destroy", t.get("name", "")))
    )


class MCPCallResult(BaseModel):
    text: str
    is_error: bool = False
    files: list[File] = Field(default_factory=list)
    auth_url: str | None = None
    """Set when the server needs the user to sign in first."""


def call_result(r: Mapping[str, Any]) -> MCPCallResult:
    """A `tools/call` result → text (plus images as files)."""
    parts: list[str] = []
    files: list[File] = []
    for p in r.get("content") or []:
        t = p.get("type")
        if t == "text":
            parts.append(p.get("text") or "")
        elif t == "image" and p.get("data"):
            mime = p.get("mimeType") or "image/png"
            files.append(File(media_type=mime, data=f"data:{mime};base64,{p['data']}"))
            parts.append(f"[image {mime} returned by the tool]")
        elif t == "resource":
            res = p.get("resource") or {}
            if res.get("text") is not None:
                parts.append(res["text"])
            elif res.get("blob") and str(res.get("mimeType", "")).startswith("image/"):
                files.append(
                    File(
                        media_type=res["mimeType"],
                        data=base64.b64decode(res["blob"]),
                    )
                )
                parts.append(f"[resource {res.get('uri')}]")
            else:
                parts.append(f"[resource {res.get('uri')}]")
        else:
            parts.append(f"[{t}]")
    if r.get("structuredContent") and not parts:
        parts.append(json.dumps(r["structuredContent"], indent=2))
    return MCPCallResult(
        text="\n".join(parts) or "(no content)",
        is_error=bool(r.get("isError")),
        files=files,
    )


class MCPSession(abc.ABC):
    """One initialized MCP session."""

    def __init__(self, client_name: str = "agentia", client_version: str = "1") -> None:
        self.protocol_version: str | None = None
        self.server_info: dict[str, Any] = {}
        self._ids = itertools.count(1)
        self._client_info = {"name": client_name, "version": client_version}

    @abc.abstractmethod
    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float = 60,
    ) -> dict[str, Any]: ...

    @abc.abstractmethod
    async def notify(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None: ...

    @abc.abstractmethod
    async def close(self) -> None: ...

    async def connect(self, timeout_s: float = 15) -> None:
        r = await self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": self._client_info,
            },
            timeout_s=timeout_s,
        )
        self.protocol_version = r.get("protocolVersion") or PROTOCOL_VERSION
        self.server_info = r.get("serverInfo") or {}
        await self.notify("notifications/initialized")

    async def list_tools(
        self, cursor: str | None = None, timeout_s: float = 15
    ) -> dict[str, Any]:
        return await self.request(
            "tools/list", {"cursor": cursor} if cursor else {}, timeout_s=timeout_s
        )

    async def list_all_tools(self, timeout_s: float = 15) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor = None
        while True:
            r = await self.list_tools(cursor, timeout_s)
            tools += [t for t in r.get("tools") or [] if isinstance(t, dict)]
            cursor = r.get("nextCursor")
            if not cursor:
                return tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any], timeout_s: float = 300
    ) -> dict[str, Any]:
        return await self.request(
            "tools/call", {"name": name, "arguments": arguments}, timeout_s=timeout_s
        )


class MCPClient(MCPSession):
    """Streamable HTTP. `send` lets the host route requests (custom transports, auth)."""

    def __init__(
        self,
        url: str,
        send: Send | httpx.AsyncClient | None = None,
        headers: dict[str, str] | None = None,
        *,
        client_name: str = "agentia",
        client_version: str = "1",
    ) -> None:
        super().__init__(client_name, client_version)
        self.url = url
        self._own: httpx.AsyncClient | None = None
        if send is None:
            self._own = httpx.AsyncClient()
            send = self._own
        self.send: Send = send.send if isinstance(send, httpx.AsyncClient) else send
        self.headers = dict(headers or {})
        self.session_id: str | None = None

    def _request(self, body: Any, timeout_s: float) -> httpx.Request:
        h = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
            **self.headers,
        }
        if self.session_id:
            h["mcp-session-id"] = self.session_id
        if self.protocol_version:
            h["mcp-protocol-version"] = self.protocol_version
        return httpx.Request(
            "POST",
            self.url,
            headers=h,
            content=json.dumps(body).encode(),
            extensions={"timeout": httpx.Timeout(timeout_s, connect=15).as_dict()},
        )

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        res = await self.send(self._request(msg, 15))
        if res.status_code == 401:
            raise Unauthorized(res.headers.get("www-authenticate"))

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float = 60,
    ) -> dict[str, Any]:
        id = next(self._ids)
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": id, "method": method}
        if params is not None:
            msg["params"] = params
        res = await self.send(self._request(msg, timeout_s))
        if res.status_code == 401:
            raise Unauthorized(res.headers.get("www-authenticate"))
        if res.status_code >= 400:
            raise MCPError(f"HTTP {res.status_code}: {res.text[:300]}")
        if sid := res.headers.get("mcp-session-id"):
            self.session_id = sid
        for m in _messages(res):
            if isinstance(m, dict) and m.get("id") == id:
                if m.get("error"):
                    err = m["error"]
                    raise MCPError(f"{err.get('message', 'error')} ({err.get('code')})")
                return m.get("result") or {}
        raise MCPError(f"no response to {method}")

    async def close(self) -> None:
        if self.session_id:
            h = {**self.headers, "mcp-session-id": self.session_id}
            with contextlib.suppress(Exception):
                await self.send(
                    httpx.Request(
                        "DELETE",
                        self.url,
                        headers=h,
                        extensions={"timeout": httpx.Timeout(5).as_dict()},
                    )
                )
            self.session_id = None
        if self._own:
            await self._own.aclose()
            self._own = None


class StdioClient(MCPSession):
    """A local MCP server run as a subprocess (newline-delimited JSON-RPC)."""

    def __init__(
        self,
        command: str,
        args: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        *,
        client_name: str = "agentia",
        client_version: str = "1",
    ) -> None:
        super().__init__(client_name, client_version)
        self.command = command
        self.args = list(args)
        self.env = dict(env or {})
        self.cwd = cwd
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._waiting: dict[int, asyncio.Future[dict[str, Any]]] = {}

    async def _start(self) -> asyncio.subprocess.Process:
        if self._proc and self._proc.returncode is None:
            return self._proc
        self._proc = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env={**os.environ, **self.env},
            cwd=self.cwd,
            limit=64 * 1024 * 1024,
        )
        self._reader = asyncio.get_running_loop().create_task(self._read(self._proc))
        return self._proc

    async def _read(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        try:
            while line := await proc.stdout.readline():
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue  # servers sometimes log to stdout
                for m in msg if isinstance(msg, list) else [msg]:
                    await self._dispatch(m)
        finally:
            for f in self._waiting.values():
                if not f.done():
                    f.set_exception(MCPError("MCP server exited"))
            self._waiting.clear()

    async def _dispatch(self, m: Any) -> None:
        if not isinstance(m, dict):
            return
        if "method" in m and "id" in m:
            # A server→client request: we offer no client capabilities.
            reply: dict[str, Any] = {"jsonrpc": "2.0", "id": m["id"]}
            if m["method"] == "ping":
                reply["result"] = {}
            else:
                reply["error"] = {"code": -32601, "message": "method not found"}
            await self._write(reply)
        elif (f := self._waiting.pop(m.get("id"), None)) and not f.done():  # type: ignore[arg-type]
            f.set_result(m)

    async def _write(self, msg: dict[str, Any]) -> None:
        proc = await self._start()
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(msg) + "\n").encode())
        await proc.stdin.drain()

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        await self._write(msg)

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float = 60,
    ) -> dict[str, Any]:
        id = next(self._ids)
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": id, "method": method}
        if params is not None:
            msg["params"] = params
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._waiting[id] = fut
        await self._write(msg)
        try:
            m = await asyncio.wait_for(fut, timeout_s)
        except TimeoutError:
            self._waiting.pop(id, None)
            raise MCPError(f"{method} timed out after {timeout_s}s")
        if m.get("error"):
            err = m["error"]
            raise MCPError(f"{err.get('message', 'error')} ({err.get('code')})")
        return m.get("result") or {}

    async def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc and proc.returncode is None:
            if proc.stdin:
                proc.stdin.close()
            try:
                await asyncio.wait_for(proc.wait(), 2)
            except TimeoutError:
                proc.kill()
                await proc.wait()
        if self._reader:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader
            self._reader = None


def _messages(res: httpx.Response) -> list[Any]:
    ct = res.headers.get("content-type", "")
    if "text/event-stream" in ct:
        out = []
        data: list[str] = []
        for line in [*res.text.split("\n"), ""]:
            line = line.removesuffix("\r")
            if not line:
                if data:
                    try:
                        out.append(json.loads("\n".join(data)))
                    except ValueError:
                        pass
                data = []
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
        return [x for m in out for x in (m if isinstance(m, list) else [m])]
    if not res.content:
        return []
    body = res.json()
    return body if isinstance(body, list) else [body]


def resource_metadata_url(www_authenticate: str | None) -> str | None:
    if not www_authenticate:
        return None
    m = re.search(r'resource_metadata="([^"]+)"', www_authenticate)
    return m.group(1) if m else None
