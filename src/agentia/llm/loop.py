"""The agent loop.

One `Agent.run()` is a sequence of steps over the history, which is the only state:

1. If the context has function calls without outputs, run them (in parallel, except
   calls sharing an `exclusive()` key). Each output is appended as soon as it exists.
   A tool may `Suspend`: its call stays pending and the run ends as `suspended`.
2. Otherwise call the model: ingest new inputs (`Agent.poll_inputs()`), let the host
   veto (`Agent.before_model_call()` may raise `StopRun`), compact if the context is
   over the threshold, then stream one response and append its items.
3. Stop when the model answers without tool calls and no new input arrived.

Because every item is appended as soon as it exists, an interrupted run (crash,
cancellation, suspension) resumes with `agent.run(None)`.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import httpx

from agentia.llm import LLMOptions
from agentia.llm.providers import ModelRequest
from agentia.models import *
from agentia.tools.tools import Suspend, to_output

if TYPE_CHECKING:
    from agentia.agent import Agent

_LOGGER = logging.getLogger("agentia.loop")


class StopRun(Exception):
    """Raised by `Agent.before_model_call()` to end the run.

    `message`, if given, is appended as the assistant's final message. The run ends with
    finish reason `stopped`.
    """

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or "stopped")
        self.message = message


class RunState(Protocol):
    usage: Usage
    last_usage: Usage
    finish_reason: FinishReason | None
    new_items: list[Item]


@dataclass
class StepInfo:
    """Passed to `Agent.before_model_call()`."""

    step: int
    """Model calls made so far in this run."""
    options: LLMOptions
    items: Sequence[Item] = field(default_factory=list)


class _Seq:
    def __init__(self) -> None:
        self.n = 0

    def __call__(self) -> int:
        self.n += 1
        return self.n


def _assign_call_ids(agent: "Agent", items: list[Item]) -> list[Item]:
    """Drop nameless calls and give every call a unique id within the history.

    Providers may omit call ids or reuse them ("call_0" every step); a duplicate would
    pair a later output with an earlier call.
    """
    seen = {i.call_id for i in agent.history.items if isinstance(i, FunctionCall)}
    out: list[Item] = []
    for item in items:
        if isinstance(item, FunctionCall):
            if not item.name:
                continue
            new_id = agent.new_call_id(item, seen)
            if new_id != item.call_id:
                item = item.model_copy(update={"call_id": new_id})
            seen.add(item.call_id)
        out.append(item)
    return out


async def _run_tools(
    agent: "Agent",
    calls: list[FunctionCall],
    options: LLMOptions,
    run: RunState,
    seq: _Seq,
) -> AsyncGenerator[StreamEvent | Suspend, None]:
    """Run pending calls; yield each recorded output's event, or a `Suspend`."""
    tools = agent.tools
    groups: dict[str, list[FunctionCall]] = {}
    for c in calls:
        key = tools.exclusive_key(c) if options.parallel_tool_calls else "sequential"
        groups.setdefault(key or f"call:{c.call_id}", []).append(c)
    queue: asyncio.Queue[StreamEvent | Suspend | BaseException | None] = asyncio.Queue()
    # Recording is serialized: hooks see outputs one at a time, in completion order.
    record_lock = asyncio.Lock()

    async def one(call: FunctionCall) -> None:
        tool = tools.get(call.name)
        r = await agent.before_tool_call(call, tool)
        if r is None:
            r = await tools.execute(call)
        if isinstance(r, Suspend):
            await queue.put(r)
            return
        out = r if isinstance(r, FunctionCallOutput) else to_output(call, r)
        async with record_lock:
            out = await agent.after_tool_call(call, out)
            agent.history.add(out)
            run.new_items.append(out)
        await queue.put(InputItemAddedEvent(sequence_number=seq(), item=out))

    async def group(g: list[FunctionCall]) -> None:
        try:
            for c in g:
                await one(c)
        except BaseException as e:  # noqa: BLE001 -- forwarded to the loop
            await queue.put(e)
        finally:
            await queue.put(None)

    tasks = [asyncio.create_task(group(g)) for g in groups.values()]
    try:
        remaining = len(tasks)
        while remaining:
            x = await queue.get()
            if x is None:
                remaining -= 1
            elif isinstance(x, BaseException):
                raise x
            else:
                yield x
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _maybe_compact(
    agent: "Agent", request: ModelRequest, seq: _Seq
) -> AsyncGenerator[StreamEvent, None]:
    from agentia.compaction import compact, context_tokens

    opts = agent.compaction
    if not opts or not opts.auto:
        return
    tokens = context_tokens(
        [i.text for i in request.instructions], request.summary, request.items
    )
    if tokens <= opts.threshold_tokens(await agent.context_window()):
        return
    yield CompactionStartedEvent(sequence_number=seq())
    try:
        c = await compact(agent, options=opts)
    except Exception:  # carry on with the full context
        _LOGGER.exception("automatic compaction failed")
        return
    if c:
        yield CompactionCompletedEvent(sequence_number=seq(), compaction=c)


async def run_loop(
    agent: "Agent",
    prompt: InputItem | None,
    options: LLMOptions,
    run: RunState,
) -> AsyncGenerator[StreamEvent, None]:
    seq = _Seq()
    yield RunStartedEvent(sequence_number=seq())
    if prompt is not None:
        items = await agent.commands.process_items(prompt)
        if not items:
            # A slash command that produced no input: nothing to send to the model.
            run.finish_reason = "stop"
            yield RunCompletedEvent(sequence_number=seq(), finish_reason="stop")
            return
        agent.history.add(*items)

    await agent.tools.init()
    finish: FinishReason = "stop"
    steps = 0
    own_client = agent.http_client is None
    client = agent.http_client or httpx.AsyncClient()
    try:
        while True:
            # 1. Pending tool calls (also how an interrupted run resumes).
            if pending := agent.history.pending_tool_calls():
                suspended = False
                async for ev in _run_tools(agent, pending, options, run, seq):
                    if isinstance(ev, Suspend):
                        suspended = True
                    else:
                        yield ev
                if suspended:
                    finish = "suspended"
                    break
                continue

            # 2. A model call.
            try:
                for item in await agent.poll_inputs():
                    agent.history.add(item)
                    run.new_items.append(item)
                    yield InputItemAddedEvent(sequence_number=seq(), item=item)
                await agent.before_model_call(
                    StepInfo(step=steps, options=options, items=agent.history.context())
                )
            except StopRun as s:
                if s.message:
                    msg = Message.assistant(s.message)
                    msg.status = "completed"
                    agent.history.add(msg)
                    run.new_items.append(msg)
                    yield InputItemAddedEvent(sequence_number=seq(), item=msg)
                finish = "stopped"
                break
            if options.max_steps is not None and steps >= options.max_steps:
                finish = "max-steps"
                break

            request = await agent.build_request(options)
            compacted = False
            async for ev in _maybe_compact(agent, request, seq):
                compacted = compacted or ev.type == "agentia.compaction.completed"
                yield ev
            if compacted:
                request = await agent.build_request(options)

            info: ResponseInfo | None = None
            async for ev in agent.provider.stream(request, client):
                ev.sequence_number = seq()
                if isinstance(ev, (ResponseCompletedEvent, ResponseIncompleteEvent)):
                    info = ev.response
                yield ev
            assert info is not None, "the provider ended without a terminal event"
            steps += 1
            run.usage += info.usage
            run.last_usage = info.usage
            output = _assign_call_ids(agent, list(info.output))
            agent.history.add(*output)
            run.new_items.extend(output)
            await agent.on_response(output, info)

            if any(isinstance(i, FunctionCall) for i in output):
                continue
            # 3. Steering: input that arrived while the model answered keeps the run going.
            if more := await agent.poll_inputs(after_answer=True):
                for item in more:
                    agent.history.add(item)
                    run.new_items.append(item)
                    yield InputItemAddedEvent(sequence_number=seq(), item=item)
                continue
            finish = info.finish_reason
            break
    finally:
        if own_client:
            await client.aclose()
        agent.history.usage += run.usage
        if run.last_usage.total_tokens is not None:
            agent.history.current_tokens = run.last_usage.total_tokens

    run.finish_reason = finish
    yield RunCompletedEvent(
        sequence_number=seq(), usage=run.usage, finish_reason=finish
    )
    await agent.events.end_of_turn.emit()


def new_call_id(call: FunctionCall, seen: set[str]) -> str:
    """The default `Agent.new_call_id`: keep the provider's id unless empty or taken."""
    if call.call_id and call.call_id not in seen:
        return call.call_id
    return f"call_{uuid.uuid4().hex}"


__all__ = ["RunState", "StepInfo", "StopRun", "new_call_id", "run_loop"]
