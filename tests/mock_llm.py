"""A scripted OpenAI Responses API behind an httpx.MockTransport, for offline tests."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from agentia import Agent
from agentia.llm.providers.openrouter import OpenRouter

BASE = "https://llm.test/api/v1"


@dataclass
class Step:
    text: str = ""
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    status: int = 0
    usage: dict[str, Any] = field(
        default_factory=lambda: {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "cost": 0.001,
        }
    )
    call_ids: list[str] | None = None


def step(
    text: str = "",
    *calls: tuple[str, dict[str, Any]],
    status: int = 0,
    call_ids: list[str] | None = None,
) -> Step:
    return Step(text=text, calls=list(calls), status=status, call_ids=call_ids)


type Script = Step | Callable[[dict[str, Any]], Step]


def sse(objs: list[Any]) -> bytes:
    return (
        "".join(f"data: {json.dumps(o)}\n\n" for o in objs) + "data: [DONE]\n\n"
    ).encode()


def events(model: str, s: Step) -> list[dict[str, Any]]:
    ev: list[dict[str, Any]] = [
        {
            "type": "response.created",
            "response": {"id": "resp_1", "model": model, "status": "in_progress"},
        }
    ]
    output: list[dict[str, Any]] = []
    idx = 0
    if s.text:
        ev.append(
            {
                "type": "response.output_item.added",
                "output_index": idx,
                "item": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                },
            }
        )
        for i in range(0, len(s.text), 5):
            ev.append(
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg_1",
                    "output_index": idx,
                    "content_index": 0,
                    "delta": s.text[i : i + 5],
                }
            )
        item = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": s.text, "annotations": []}],
        }
        ev.append(
            {"type": "response.output_item.done", "output_index": idx, "item": item}
        )
        output.append(item)
        idx += 1
    for k, (name, args) in enumerate(s.calls):
        cid = s.call_ids[k] if s.call_ids else f"call_{k}"
        item = {
            "id": f"fc_{k}",
            "type": "function_call",
            "call_id": cid,
            "name": name,
            "arguments": "",
        }
        ev.append(
            {"type": "response.output_item.added", "output_index": idx, "item": item}
        )
        ev.append(
            {
                "type": "response.function_call_arguments.delta",
                "item_id": f"fc_{k}",
                "output_index": idx,
                "delta": json.dumps(args),
            }
        )
        done = {**item, "arguments": json.dumps(args), "status": "completed"}
        ev.append(
            {"type": "response.output_item.done", "output_index": idx, "item": done}
        )
        output.append(done)
        idx += 1
    ev.append(
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": model,
                "status": "completed",
                "output": output,
                "usage": s.usage,
            },
        }
    )
    return ev


class MockLLM:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.script: list[Script] = []
        self.fallback: Callable[[dict[str, Any]], Step] = lambda r: Step(text="ok")
        self.models = [{"id": "test/model", "context_length": 8000}]

    def handler(self, req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": self.models})
        body = json.loads(req.content)
        self.requests.append(body)
        s = self.script.pop(0) if self.script else self.fallback
        st = s if isinstance(s, Step) else s(body)
        if st.status:
            return httpx.Response(
                st.status, json={"error": {"message": f"HTTP {st.status}"}}
            )
        return httpx.Response(
            200,
            content=sse(events(body["model"], st)),
            headers={"content-type": "text/event-stream"},
        )

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))

    def provider(self, model: str = "test/model") -> OpenRouter:
        return OpenRouter(model, api_key="k", base_url=BASE, client=self.client())

    def agent(self, cls: type[Agent] = Agent, **kw: Any) -> Agent:
        client = self.client()
        return cls(
            provider=OpenRouter(
                kw.pop("model", "test/model"), api_key="k", base_url=BASE, client=client
            ),
            client=client,
            **kw,
        )


def last_user(r: dict[str, Any]) -> str:
    users = [
        i for i in r["input"] if i.get("type") == "message" and i.get("role") == "user"
    ]
    if not users:
        return ""
    c = users[-1]["content"]
    return c if isinstance(c, str) else "".join(p.get("text", "") for p in c)


def outputs(r: dict[str, Any]) -> list[str]:
    return [
        i["output"] if isinstance(i["output"], str) else str(i["output"])
        for i in r["input"]
        if i.get("type") == "function_call_output"
    ]


def system_text(r: dict[str, Any]) -> str:
    parts = [r.get("instructions") or ""]
    for i in r["input"]:
        if i.get("type") == "message" and i.get("role") == "system":
            parts += [p.get("text", "") for p in i["content"]]
    return "\n\n".join(parts)
