"""The agent loop, offline: tools, suspension and resumption, steering, hooks."""

import asyncio
from typing import Annotated, Any

from pydantic import BaseModel

from agentia import (
    Agent,
    FunctionCall,
    FunctionCallOutput,
    Message,
    ModelTool,
    StepInfo,
    StopRun,
    Suspend,
    ToolCall,
    ToolResult,
)
from agentia.models import Item

from .mock_llm import MockLLM, last_user, outputs, step


def add(a: Annotated[int, "first"], b: Annotated[int, "second"]) -> int:
    """Add two numbers"""
    return a + b


async def test_plain_answer():
    llm = MockLLM()
    llm.script += [step("hello there")]
    agent = llm.agent()
    msg = await agent.run("hi")
    assert msg.text == "hello there"
    assert [i.type for i in agent.history.items] == ["message", "message"]
    assert agent.usage.cost == 0.001
    assert agent.current_context_length == 15


async def test_tool_call_then_answer():
    llm = MockLLM()
    llm.script += [
        step("", ("add", {"a": 2, "b": 3})),
        lambda r: step(f"sum={outputs(r)[-1]}"),
    ]
    agent = llm.agent(tools=[add])
    run = agent.run("add")
    items = [i async for i in run]
    assert [i.type for i in items] == [
        "function_call",
        "function_call_output",
        "message",
    ]
    assert agent.history.items[-1].text == "sum=5"  # type: ignore[union-attr]
    assert run.finish_reason == "stop"
    assert llm.requests[0]["tools"][0]["name"] == "add"


async def test_tool_errors_are_reported_not_raised():
    llm = MockLLM()

    def boom() -> str:
        raise RuntimeError("kaboom")

    llm.script += [
        step("", ("boom", {}), ("missing", {}), ("add", {"a": "x"})),
        lambda r: step("|".join(outputs(r))),
    ]
    msg = await llm.agent(tools=[boom, add]).run("go")
    assert (
        "kaboom" in msg.text
        and "unknown tool 'missing'" in msg.text
        and "Error parsing argument" in msg.text
    )


async def test_duplicate_call_ids_are_renamed():
    llm = MockLLM()
    llm.script += [
        step("", ("add", {"a": 1, "b": 1}), call_ids=["call_0"]),
        step("", ("add", {"a": 2, "b": 2}), call_ids=["call_0"]),
        step("done"),
    ]
    agent = llm.agent(tools=[add])
    await agent.run("go")
    ids = [i.call_id for i in agent.history.items if isinstance(i, FunctionCall)]
    assert len(set(ids)) == 2
    outs = {
        i.call_id: i.text
        for i in agent.history.items
        if isinstance(i, FunctionCallOutput)
    }
    assert sorted(outs.values()) == ["2", "4"]


class _PathArgs(BaseModel):
    path: str


async def test_exclusive_calls_run_in_order_others_in_parallel():
    llm = MockLLM()
    running: list[str] = []
    peak = [0]
    order: list[str] = []

    async def write(ctx: ToolCall, a: _PathArgs) -> str:
        running.append(a.path)
        peak[0] = max(peak[0], len(running))
        await asyncio.sleep(0.01)
        order.append(ctx.call_id)
        running.remove(a.path)
        return "ok"

    tool = ModelTool("write", "w", _PathArgs, write, exclusive=lambda a: a.path)
    llm.script += [
        step(
            "",
            ("write", {"path": "a"}),
            ("write", {"path": "a"}),
            ("write", {"path": "b"}),
            call_ids=["c1", "c2", "c3"],
        ),
        step("done"),
    ]
    await llm.agent(tools=[tool]).run("go")
    assert peak[0] == 2  # a and b concurrently, never a twice
    assert order.index("c1") < order.index("c2")


async def test_suspend_and_resume_with_an_external_output():
    llm = MockLLM()

    async def wait_for_child(ctx: ToolCall) -> Any:
        return Suspend("waiting")

    llm.script += [
        step("", ("wait_for_child", {}), ("add", {"a": 1, "b": 2})),
        lambda r: step("|".join(outputs(r))),
    ]
    agent = llm.agent(tools=[wait_for_child, add])
    run = agent.run("go")
    await run
    assert run.finish_reason == "suspended"
    assert [c.name for c in agent.history.pending_tool_calls()] == ["wait_for_child"]
    pending = agent.history.pending_tool_calls()[0]
    agent.history.add(
        FunctionCallOutput.create(call_id=pending.call_id, output="child report")
    )
    msg = await agent.run(None)
    # Outputs are paired with their calls (in call order) whatever order they were added in.
    assert msg.text == "child report|3"


async def test_resume_after_a_crash_runs_only_the_pending_call():
    llm = MockLLM()
    calls: list[int] = []

    def count(n: int) -> int:
        calls.append(n)
        return n

    llm.script += [lambda r: step(",".join(outputs(r)))]
    agent = llm.agent(tools=[count])
    # Simulate a crash: the response was recorded and one of two calls finished.
    agent.history.add(Message.user("go"))
    agent.history.add(
        FunctionCall.create(call_id="a", name="count", input={"n": 1}),
        FunctionCall.create(call_id="b", name="count", input={"n": 2}),
        FunctionCallOutput.create(call_id="a", output=1),
    )
    msg = await agent.run(None)
    assert calls == [2]
    assert msg.text == "1,2"
    assert len(llm.requests) == 1


class Approving(Agent):
    approved: set[str]
    asked: list[str]

    async def before_tool_call(self, call, tool):
        if tool and tool.name == "rm" and call.call_id not in self.approved:
            self.asked.append(call.call_id)
            return Suspend("approval")
        return None


async def test_before_tool_call_can_park_a_call_for_approval():
    llm = MockLLM()

    def rm(path: str) -> str:
        return f"removed {path}"

    llm.script += [step("", ("rm", {"path": "/x"})), lambda r: step(outputs(r)[-1])]
    agent = llm.agent(Approving, tools=[rm])
    assert isinstance(agent, Approving)
    agent.approved, agent.asked = set(), []
    run = agent.run("clean")
    await run
    assert run.suspended and len(agent.asked) == 1
    agent.approved.add(agent.asked[0])
    assert (await agent.run(None)).text == "removed /x"


class Steered(Agent):
    inbox: list[str]
    seen_steps: list[int]

    async def poll_inputs(self, after_answer: bool = False):
        items: list[Item] = [Message.user(t) for t in self.inbox]
        self.inbox.clear()
        return items

    async def before_model_call(self, step: StepInfo) -> None:
        self.seen_steps.append(step.step)
        if step.step >= 3:
            raise StopRun("⚠️ step limit")


async def test_input_arriving_during_the_answer_continues_the_run():
    llm = MockLLM()
    agent = llm.agent(Steered)
    assert isinstance(agent, Steered)
    agent.inbox, agent.seen_steps = [], []

    def first(r: dict[str, Any]):
        agent.inbox.append("also this")
        return step("first answer")

    llm.script += [first, lambda r: step(f"saw: {last_user(r)}")]
    run = agent.run("hello")
    msg = await run
    assert msg.text == "saw: also this"
    assert run.finish_reason == "stop"


async def test_stop_run_appends_its_message():
    llm = MockLLM()
    llm.fallback = lambda r: step("", ("add", {"a": 1, "b": 1}))
    agent = llm.agent(Steered, tools=[add])
    assert isinstance(agent, Steered)
    agent.inbox, agent.seen_steps = [], []
    run = agent.run("loop forever")
    msg = await run
    assert run.finish_reason == "stopped" and msg.text == "⚠️ step limit"
    assert agent.seen_steps == [0, 1, 2, 3]


async def test_max_steps():
    llm = MockLLM()
    llm.fallback = lambda r: step("", ("add", {"a": 1, "b": 1}))
    agent = llm.agent(tools=[add])
    from agentia import LLMOptions

    run = agent.run("go", options=LLMOptions(max_steps=2))
    await run
    assert run.finish_reason == "max-steps" and len(llm.requests) == 2


class Recording(Agent):
    log: list[str]

    async def on_response(self, items, response):
        self.log.append(f"response:{response.model}:{len(items)}")

    async def after_tool_call(self, call, output):
        self.log.append(f"after:{call.name}")
        return output.model_copy(
            update={"output": f"[{output.text}]", "meta": {"short": True}}
        )


async def test_after_tool_call_can_replace_the_output():
    llm = MockLLM()
    llm.script += [step("", ("add", {"a": 1, "b": 1})), lambda r: step(outputs(r)[-1])]
    agent = llm.agent(Recording, tools=[add])
    assert isinstance(agent, Recording)
    agent.log = []
    assert (await agent.run("go")).text == "[2]"
    out = next(i for i in agent.history.items if isinstance(i, FunctionCallOutput))
    assert out.meta == {"short": True}
    assert agent.log == ["response:test/model:1", "after:add", "response:test/model:1"]


async def test_tool_results_with_meta_and_files():
    llm = MockLLM()

    def shot() -> ToolResult:
        from agentia import File

        return ToolResult(
            output="see image",
            files=[File(media_type="image/png", data=b"\x89PNG")],
            meta={"k": 1},
        )

    llm.script += [step("", ("shot", {})), step("ok")]
    agent = llm.agent(tools=[shot])
    await agent.run("go")
    out = next(i for i in agent.history.items if isinstance(i, FunctionCallOutput))
    assert out.meta == {"k": 1}
    sent = next(
        i for i in llm.requests[1]["input"] if i["type"] == "function_call_output"
    )
    assert [p["type"] for p in sent["output"]] == ["input_text", "input_image"]


async def test_streaming_events_and_run_stream():
    llm = MockLLM()
    llm.script += [step("", ("add", {"a": 1, "b": 2})), step("three")]
    agent = llm.agent(tools=[add])
    types = [e.type async for e in agent.run("go", stream=True, events=True)]
    assert types[0] == "agentia.run.started" and types[-1] == "agentia.run.completed"
    assert "agentia.input_item.added" in types and "response.output_text.delta" in types

    llm.script += [step("", ("add", {"a": 1, "b": 2})), step("three")]
    got: list[str] = []
    async for part in agent.run("again", stream=True):
        if isinstance(part, (FunctionCall, FunctionCallOutput)):
            got.append(part.type)
        else:
            got.append(f"text:{await part}")
    assert got == ["function_call", "function_call_output", "text:three"]


async def test_cancellation_leaves_the_history_resumable():
    llm = MockLLM()
    started = asyncio.Event()

    async def slow() -> str:
        started.set()
        await asyncio.sleep(10)
        return "never"

    llm.script += [step("", ("slow", {}))]
    agent = llm.agent(tools=[slow])
    task = asyncio.create_task(_consume(agent))
    await started.wait()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert [c.name for c in agent.history.pending_tool_calls()] == ["slow"]


async def _consume(agent: Agent) -> None:
    await agent.run("go")


def test_per_run_options_only_override_what_they_set():
    from agentia import LLMOptions

    llm = MockLLM()
    agent = llm.agent(options=LLMOptions(parallel_tool_calls=False, temperature=0.5))
    merged = agent._Agent__merge_options(LLMOptions(max_steps=3))  # type: ignore[attr-defined]
    assert (
        merged.parallel_tool_calls is False
        and merged.temperature == 0.5
        and merged.max_steps == 3
    )
