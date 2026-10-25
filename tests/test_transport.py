"""The Responses API transport, offline: SSE parsing, errors, retries, fallbacks, caching."""

import json
from typing import Any

import httpx
import pytest

from agentia import Instruction, LLMError, LLMOptions, Message
from agentia.llm.providers import ModelRequest
from agentia.llm.providers._openai_api import StreamAccumulator, iter_sse
from agentia.llm.providers.openrouter import OpenRouter
from agentia.models import FunctionCall

from .mock_llm import BASE, MockLLM, step


def sse(o: Any) -> str:
    return f"data: {json.dumps(o)}\n\n"


def decode(raw: str, reasoning: bool = False) -> tuple[list[Any], StreamAccumulator]:
    acc = StreamAccumulator(reasoning)
    out: list[Any] = []
    feed = iter_sse(lambda d: out.extend(acc.event(json.loads(d))))
    for i in range(0, len(raw), 7):  # arbitrary chunking
        feed(raw[i : i + 7])
    feed(None)
    out += acc.end()
    return out, acc


def test_assembles_text_and_fragmented_function_calls_without_done_events():
    raw = (
        ": OPENROUTER PROCESSING\n\n"
        "event: response.created\n"
        + sse(
            {
                "type": "response.created",
                "response": {"id": "r", "model": "m", "status": "in_progress"},
            }
        )
        + sse(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "msg",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                },
            }
        )
        + sse(
            {
                "type": "response.output_text.delta",
                "item_id": "msg",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hel",
            }
        )
        + sse(
            {
                "type": "response.output_text.delta",
                "item_id": "msg",
                "output_index": 0,
                "content_index": 0,
                "delta": "lo",
            }
        )
        + sse(
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "id": "fc1",
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "bash",
                    "arguments": "",
                },
            }
        )
        + sse(
            {
                "type": "response.output_item.added",
                "output_index": 2,
                "item": {
                    "id": "fc2",
                    "type": "function_call",
                    "call_id": "c2",
                    "name": "read_file",
                    "arguments": "",
                },
            }
        )
        + sse(
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc1",
                "output_index": 1,
                "delta": '{"cm',
            }
        )
        + sse(
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc2",
                "output_index": 2,
                "delta": '{"path":"a"}',
            }
        )
        + sse(
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc1",
                "output_index": 1,
                "delta": 'd":"ls"}',
            }
        )
        + sse(
            {
                "type": "response.completed",
                "response": {
                    "model": "m",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cost": 0.001,
                        "input_tokens_details": {"cached_tokens": 4},
                    },
                },
            }
        )
        + "data: [DONE]\n\n"
    )
    events, _ = decode(raw)
    assert (
        "".join(e.delta for e in events if e.type == "response.output_text.delta")
        == "Hello"
    )
    done = events[-1]
    assert done.type == "response.completed"
    out = done.response.output
    assert out[0].text == "Hello"
    assert [(c.call_id, c.name, c.arguments) for c in out[1:]] == [
        ("c1", "bash", '{"cmd":"ls"}'),
        ("c2", "read_file", '{"path":"a"}'),
    ]
    assert done.response.finish_reason == "tool-calls" and done.response.model == "m"
    u = done.response.usage
    assert (u.input_tokens, u.output_tokens, u.cached_input_tokens, u.cost) == (
        10,
        5,
        4,
        0.001,
    )


def test_the_completed_responses_output_is_authoritative():
    events, _ = decode(
        sse(
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "final text"}],
                        },
                        {
                            "type": "function_call",
                            "call_id": "x1",
                            "name": "grep",
                            "arguments": '{"pattern":"a"}',
                        },
                    ],
                },
            }
        )
    )
    out = events[-1].response.output
    assert (
        out[0].text == "final text"
        and isinstance(out[1], FunctionCall)
        and out[1].call_id == "x1"
    )


def test_a_stream_without_a_terminal_event_still_completes():
    events, _ = decode(
        sse(
            {
                "type": "response.output_text.delta",
                "item_id": "m",
                "output_index": 0,
                "delta": "partial",
            }
        )
    )
    assert (
        events[-1].type == "response.completed"
        and events[-1].response.output[0].text == "partial"
    )


def test_reasoning_is_dropped_unless_enabled():
    raw = sse(
        {
            "type": "response.reasoning_text.delta",
            "item_id": "rs",
            "output_index": 0,
            "delta": "hmm",
        }
    ) + sse(
        {
            "type": "response.completed",
            "response": {
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs",
                        "content": [{"type": "reasoning_text", "text": "hmm"}],
                    }
                ]
            },
        }
    )
    off, _ = decode(raw, reasoning=False)
    assert [e.type for e in off] == ["response.completed"] and off[
        -1
    ].response.output == []
    on, _ = decode(raw, reasoning=True)
    assert (
        on[0].type == "response.reasoning_text.delta"
        and on[-1].response.output[0].text == "hmm"
    )


def test_provider_errors_mid_stream_raise():
    with pytest.raises(LLMError):
        decode(sse({"type": "error", "error": {"message": "overloaded", "code": 502}}))
    with pytest.raises(LLMError, match="boom"):
        decode(
            sse(
                {
                    "type": "response.failed",
                    "response": {"error": {"message": "boom", "code": "server_error"}},
                }
            )
        )
    with pytest.raises(LLMError):
        decode(sse({"error": {"message": "overloaded", "code": 502}}))
    with pytest.raises(LLMError, match="empty stream"):
        decode(": keepalive\n\n")


async def collect(provider: OpenRouter, request: ModelRequest) -> list[Any]:
    return [e async for e in provider.stream(request)]


REQ = ModelRequest(items=[Message.user("x")], options=LLMOptions(max_retries=2))


async def test_retries_5xx_but_not_401_and_falls_back_to_the_next_model(
    monkeypatch: pytest.MonkeyPatch,
):
    import agentia.llm.providers._openai_api as api

    async def no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr(api.asyncio, "sleep", no_sleep)
    llm = MockLLM()
    llm.script += [step(status=503), step("recovered")]
    events = await collect(llm.provider(), REQ)
    assert events[-1].response.output[0].text == "recovered"

    llm.requests.clear()
    llm.script += [step(status=401), step("never")]
    with pytest.raises(LLMError):
        await collect(
            llm.provider(),
            ModelRequest(
                items=REQ.items,
                options=LLMOptions(max_retries=2, fallback_models=["other"]),
            ),
        )
    assert len(llm.requests) == 1

    llm.requests.clear()
    llm.script[:] = [step(status=404), step("from fallback")]
    events = await collect(
        llm.provider("missing/model"),
        ModelRequest(
            items=REQ.items,
            options=LLMOptions(max_retries=0, fallback_models=["test/model"]),
        ),
    )
    assert events[-1].response.output[0].text == "from fallback"
    assert [r["model"] for r in llm.requests] == ["missing/model", "test/model"]


async def test_never_retries_once_output_streamed():
    calls = [0]

    def handler(req: httpx.Request) -> httpx.Response:
        calls[0] += 1
        body = sse(
            {
                "type": "response.output_text.delta",
                "item_id": "m",
                "output_index": 0,
                "delta": "Hi",
            }
        ) + sse({"type": "error", "error": {"message": "overloaded", "code": 502}})
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    p = OpenRouter(
        "m",
        api_key="k",
        base_url=BASE,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    seen: list[str] = []
    with pytest.raises(LLMError):
        async for e in p.stream(REQ):
            seen.append(e.type)
    assert calls[0] == 1 and seen == ["response.output_text.delta"]


async def test_json_body_fallback():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "plain"}],
                    }
                ],
            },
        )

    p = OpenRouter(
        "m",
        api_key="k",
        base_url=BASE,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    events = await collect(p, REQ)
    assert events[-1].response.output[0].text == "plain"


def test_request_body_and_cache_breakpoints():
    req = ModelRequest(
        items=[Message.user("hi")],
        instructions=[
            Instruction(text="STATIC"),
            Instruction(text="NOW", dynamic=True),
            Instruction(text="PLUGIN"),
        ],
        summary="SUMMARY",
    )
    plain = OpenRouter("openai/gpt-5", api_key="k").build_body(req, "openai/gpt-5")
    assert plain["instructions"] == "STATIC\n\nPLUGIN\n\nSUMMARY\n\nNOW"
    assert plain["store"] is False and plain["stream"] is True and "tools" not in plain

    cached = OpenRouter("anthropic/claude-x", api_key="k").build_body(
        req, "anthropic/claude-x"
    )
    assert "instructions" not in cached
    sys_msg = cached["input"][0]
    assert sys_msg["role"] == "system"
    assert [(p["text"], "cache_control" in p) for p in sys_msg["content"]] == [
        ("STATIC", False),
        ("PLUGIN", True),
        ("SUMMARY", True),
        ("NOW", False),
    ]
    assert cached["input"][1]["role"] == "user"
    forced = OpenRouter("openai/gpt-5", api_key="k").build_body(
        ModelRequest(
            items=req.items,
            instructions=req.instructions,
            options=LLMOptions(cache_control=True),
        ),
        "x",
    )
    assert forced["input"][0]["role"] == "system"


async def test_context_length_from_the_providers_own_catalog():
    llm = MockLLM()
    llm.models = [{"id": "test/ctx", "context_length": 1234}]
    assert await llm.provider("test/ctx").get_context_length() == 1234
