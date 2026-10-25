"""Skills with a custom source, and MCP over streamable HTTP and stdio, offline."""

import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import httpx

from agentia import Skill, Skills, SkillSource
from agentia.mcp import (
    MCP,
    MCPCallResult,
    MCPClient,
    MCPServerInfo,
    MCPToolInfo,
    MCPTools,
)
from agentia.mcp.tools import MCPServers

from .mock_llm import MockLLM, outputs, step, system_text


class MemorySource(SkillSource):
    def __init__(self, files: dict[str, str]) -> None:
        self.files = files

    async def list_skills(self) -> list[Skill]:
        return [
            Skill(name=p.split("/")[-2], description="d", path=p) for p in self.files
        ]

    async def read_skill(self, skill: Skill, args: list[str] | None = None) -> str:
        return self.files[skill.path]


async def test_skills_from_a_custom_source():
    llm = MockLLM()
    source = MemorySource({"/opt/skills/pdf/SKILL.md": "use pdftotext"})
    llm.script += [step("", ("skill", {"name": "pdf"})), lambda r: step(outputs(r)[-1])]
    agent = llm.agent(tools=[Skills(source=source, tool_name="skill")])
    msg = await agent.run("read the pdf")
    assert msg.text == "# Skill pdf (/opt/skills/pdf)\n\nuse pdftotext"
    s = system_text(llm.requests[0])
    assert "- pdf: d (/opt/skills/pdf/SKILL.md)" in s and "`skill` tool" in s


async def test_local_skills_and_slash_commands(tmp_path: Path):
    d = tmp_path / "weather"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: weather\ndescription: Weather\n---\nWeather Skill for $ARGUMENTS"
    )
    llm = MockLLM()
    agent = llm.agent(skills=[tmp_path])
    out = await agent.commands.handle("/weather Sydney")
    assert out and "Weather Skill" in out


# ---------------------------------------------------------------- MCP over HTTP


class FakeMcp:
    def __init__(self, tools: list[dict[str, Any]], needs_auth: bool = False) -> None:
        self.tools = tools
        self.needs_auth = needs_auth
        self.calls: list[dict[str, Any]] = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        if self.needs_auth and req.headers.get("authorization") != "Bearer tok":
            return httpx.Response(
                401,
                headers={
                    "www-authenticate": 'Bearer resource_metadata="https://mcp.test/.well-known/x"'
                },
            )
        if req.method == "DELETE":
            return httpx.Response(200)
        msg = json.loads(req.content)
        if "id" not in msg:
            return httpx.Response(202)
        match msg["method"]:
            case "initialize":
                result: dict[str, Any] = {
                    "protocolVersion": "2025-06-18",
                    "serverInfo": {"name": "fake"},
                }
            case "tools/list":
                result = {"tools": self.tools}
            case "tools/call":
                self.calls.append(msg["params"])
                result = {
                    "content": [
                        {"type": "text", "text": f"called {msg['params']['name']}"}
                    ]
                }
            case _:
                result = {}
        # Answer as SSE, like many servers do.
        body = f"event: message\ndata: {json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': result})}\n\n"
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream", "mcp-session-id": "s1"},
        )

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


async def test_mcp_http_server_as_tools():
    fake = FakeMcp(
        [
            {"name": "add", "description": "Add", "inputSchema": {"type": "object"}},
            {"name": "delete_all", "inputSchema": {}},
        ]
    )
    server = MCP("calc", url="https://mcp.test/mcp", client=fake.client())
    llm = MockLLM()
    llm.script += [
        step("", ("mcp__calc__add", {"a": 1})),
        lambda r: step(outputs(r)[-1]),
    ]
    async with llm.agent(tools=[server]) as agent:
        assert (await agent.run("add")).text == "called add"
        tool = agent.tools.get("mcp__calc__delete_all")
        assert tool and tool.metadata["mcp"]["destructive"] is True
    assert fake.calls == [{"name": "add", "arguments": {"a": 1}}]
    assert server.session is None  # closed with the agent
    assert "# MCP servers\n- calc: connected, 2 tools" in system_text(llm.requests[0])


async def test_mcp_client_raises_unauthorized():
    from agentia.mcp import Unauthorized

    fake = FakeMcp([], needs_auth=True)
    c = MCPClient("https://mcp.test/mcp", fake.client())
    try:
        await c.connect()
        raise AssertionError("expected Unauthorized")
    except Unauthorized as e:
        assert "resource_metadata" in (e.www_authenticate or "")
    c2 = MCPClient(
        "https://mcp.test/mcp", fake.client(), {"authorization": "Bearer tok"}
    )
    await c2.connect()
    assert c2.session_id == "s1"


class Catalog:
    """A host-managed catalog (e.g. with OAuth): the protocol, not MCPServers."""

    def __init__(self, n: int) -> None:
        self.tools = [
            MCPToolInfo(server="big", name=f"tool{i}", description=f"does thing {i}")
            for i in range(n)
        ]

    async def catalog(self):
        return [MCPServerInfo(name="big", tools=len(self.tools))], self.tools

    async def call(
        self, server: str, tool: str, arguments: dict[str, Any]
    ) -> MCPCallResult:
        if tool == "tool3":
            return MCPCallResult(
                text="authorization required", is_error=True, auth_url="https://login"
            )
        return MCPCallResult(text=f"{server}/{tool} {json.dumps(arguments)}")


async def test_mcp_search_mode_and_auth_callback():
    auth: list[tuple[str, str]] = []
    tools = MCPTools(
        Catalog(50), on_auth_required=lambda s, u: auth.append((s, u)) or None
    )
    llm = MockLLM()
    llm.script += [
        step("", ("mcp_search", {"query": "thing 7"})),
        step(
            "", ("mcp_call", {"server": "big", "tool": "tool7", "arguments": {"x": 1}})
        ),
        step("", ("mcp_call", {"server": "big", "tool": "tool3"})),
        lambda r: step(" || ".join(outputs(r))),
    ]
    msg = await llm.agent(tools=[tools]).run("go")
    assert [t["name"] for t in llm.requests[0]["tools"]] == ["mcp_search", "mcp_call"]
    assert "big/tool7: does thing 7" in msg.text and 'big/tool7 {"x": 1}' in msg.text
    assert "needs authorization" in msg.text and auth == [("big", "https://login")]
    assert "use mcp_search to find one" in system_text(llm.requests[0])


async def test_mcp_allowlist():
    tools = MCPTools(Catalog(3), allow=["mcp__big__tool1"])
    llm = MockLLM()
    llm.script += [step("hi")]
    await llm.agent(tools=[tools]).run("go")
    assert [t["name"] for t in llm.requests[0]["tools"]] == ["mcp__big__tool1"]


# ---------------------------------------------------------------- MCP over stdio

STDIO_SERVER = textwrap.dedent(
    """
    import json, sys
    for line in sys.stdin:
        msg = json.loads(line)
        if "id" not in msg:
            continue
        m = msg["method"]
        if m == "initialize":
            r = {"protocolVersion": "2025-06-18", "serverInfo": {"name": "echo"}}
        elif m == "tools/list":
            r = {"tools": [{"name": "echo", "inputSchema": {"type": "object"}, "annotations": {"readOnlyHint": True}}]}
        elif m == "tools/call":
            r = {"content": [{"type": "text", "text": "echo " + json.dumps(msg["params"]["arguments"])}]}
        else:
            r = {}
        print("log line that is not json", flush=True)
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": r}), flush=True)
    """
)


async def test_mcp_stdio_server(tmp_path: Path):
    script = tmp_path / "server.py"
    script.write_text(STDIO_SERVER)
    servers = MCPServers(
        [MCP("echo", command=sys.executable, args=["-I", str(script)])]
    )
    try:
        infos, tools = await servers.catalog()
        assert (
            infos[0].status == "connected"
            and tools[0].name == "echo"
            and not tools[0].destructive
        )
        r = await servers.call("echo", "echo", {"x": 1})
        assert r.text == 'echo {"x": 1}'
    finally:
        await servers.aclose()
