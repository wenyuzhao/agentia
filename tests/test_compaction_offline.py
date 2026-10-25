"""History and compaction, offline."""

from typing import Any

import pytest

from agentia import (
    Agent,
    Compaction,
    CompactionOptions,
    FunctionCall,
    FunctionCallOutput,
    History,
    Message,
    normalize_tool_pairs,
)
from agentia.compaction import DEFAULT_INSTRUCTIONS, split_point
from agentia.models import Item

from .mock_llm import MockLLM, step, system_text


def exchange(n: int, steps: int) -> list[Item]:
    items: list[Item] = [Message.user(f"u{n}")]
    for i in range(steps):
        items.append(FunctionCall.create(call_id=f"c{n}_{i}", name="t"))
        items.append(FunctionCallOutput.create(call_id=f"c{n}_{i}", output="o"))
    items.append(Message.assistant(f"a{n}"))
    return items


def test_split_keeps_the_last_k_exchanges():
    items = [x for n in range(6) for x in exchange(n, 1)]
    cut = split_point(items, keep_exchanges=4)
    assert cut == 8 and items[cut].text == "u2"  # type: ignore[union-attr]


def test_split_inside_one_huge_exchange_at_a_response_boundary():
    items = exchange(0, 30)
    cut = split_point(items, keep_exchanges=4, keep_steps=6)
    assert cut > 0 and isinstance(items[cut], FunctionCall)
    recent = items[cut:]
    ids = {i.call_id for i in recent if isinstance(i, FunctionCall)}
    assert all(i.call_id in ids for i in recent if isinstance(i, FunctionCallOutput))


def test_split_returns_nothing_when_too_little():
    assert split_point(exchange(0, 1), 4) == 0
    assert split_point(exchange(0, 1), 0) == 4  # keep nothing: fold everything


def test_history_is_append_only_and_context_follows_the_summary():
    h = History()
    h.add(*exchange(0, 1), *exchange(1, 0))
    h.add_compaction(Compaction(summary="S", covers=4))
    assert len(h.items) == 6 and [i.text for i in h.context()] == ["u1", "a1"]  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        h.add_compaction(Compaction(summary="S", covers=2))
    with pytest.raises(ValueError):
        h.add_compaction(Compaction(summary="S", covers=99))


def test_normalize_tool_pairs():
    items: list[Item] = [
        Message.user("go"),
        FunctionCall.create(call_id="a", name="t"),
        FunctionCall.create(call_id="b", name="t"),
        FunctionCallOutput.create(call_id="b", output="B"),
        Message.user("steer"),
        FunctionCallOutput.create(call_id="a", output="A"),
        FunctionCallOutput.create(call_id="zzz", output="orphan"),
        FunctionCall.create(call_id="c", name="t"),
    ]
    out = normalize_tool_pairs(items)
    desc = [
        f"call:{i.call_id}" if isinstance(i, FunctionCall) else i.text  # type: ignore[union-attr]
        for i in out
    ]
    assert desc == [
        "go",
        "call:a",
        "call:b",
        "A",
        "B",
        "steer",
        "call:c",
        "[no result recorded]",
    ]


class Hooked(Agent):
    seen: list[str]

    async def on_compaction(self, compaction, replaced):
        self.seen.append(f"compaction:{compaction.covers}:{len(replaced)}")


def is_summarizer(r: dict[str, Any]) -> bool:
    return r.get("instructions") == DEFAULT_INSTRUCTIONS


async def test_auto_compaction_before_a_model_call():
    llm = MockLLM()
    llm.models = [{"id": "test/model", "context_length": 2000}]  # threshold 1200 tokens

    def fb(r: dict[str, Any]):
        if is_summarizer(r):
            return step("SUMMARY: user is Alice")
        return step("ok")

    llm.fallback = fb
    agent = llm.agent(
        Hooked, compaction=CompactionOptions(keep_exchanges=2), instructions="BASE"
    )
    assert isinstance(agent, Hooked)
    agent.seen = []
    await agent.run("my name is Alice")
    filler = "lorem ipsum " * 200
    for i in range(4):
        await agent.run(f"message {i} {filler}")
    summarizer = [r for r in llm.requests if is_summarizer(r)]
    assert summarizer, "compaction should have run"
    assert "my name is Alice" in summarizer[0]["input"][0]["content"][0]["text"]
    last = [r for r in llm.requests if not is_summarizer(r)][-1]
    assert "SUMMARY: user is Alice" in system_text(last)
    assert system_text(last).index("BASE") < system_text(last).index("SUMMARY")
    assert all("my name is Alice" not in str(i) for i in last["input"])
    c = agent.history.compaction
    assert c and c.covers > 0 and agent.seen[0].startswith("compaction:")
    # Nothing was removed from the history.
    assert agent.history.items[0].text == "my name is Alice"  # type: ignore[union-attr]


async def test_manual_compaction_with_focus_and_keep_zero():
    llm = MockLLM()
    llm.fallback = lambda r: (
        step(f"FOCUS={'yes' if 'the database' in str(r['input']) else 'no'}")
        if is_summarizer(r)
        else step("ok")
    )
    agent = llm.agent()
    await agent.run("hello")
    assert await agent.compact("the database") is None  # nothing old enough
    c = await agent.compact("the database", keep_exchanges=0)
    assert (
        c
        and c.summary == "FOCUS=yes"
        and c.covers == 2
        and agent.history.context() == []
    )
    # A summarizer model override uses the same endpoint and credentials.
    llm.models = [{"id": "test/model", "context_length": 8000}]
    await agent.run("again")
    await agent.compact(keep_exchanges=0, model="test/cheap")
    assert llm.requests[-1]["model"] == "test/cheap"
