from collections.abc import AsyncGenerator, Sequence
from typing import TYPE_CHECKING

import httpx

from agentia.history import History
from agentia.llm import LLMOptions
from agentia.llm.completion import Run
from agentia.llm.stream import RunEvents, RunStream
from agentia.models import *
from agentia.tools.tools import ToolSet

if TYPE_CHECKING:
    from agentia.agent import Agent


async def __process_tool_calls(
    agent: "Agent", tool_calls: list[FunctionCall], tools: ToolSet, parallel: bool
) -> list[FunctionCallOutput]:
    assert tools is not None, "No tools provided"
    return await tools.run(agent, tool_calls, parallel)


async def __add_prompt(agent: "Agent", history: History, prompt: InputItem) -> bool:
    """Add the prompt to history. Returns False if the prompt was a slash command
    that produced no output, in which case the LLM should be skipped."""

    prompt = await agent.commands.process_items(prompt)

    if not prompt:
        return False

    history.add(*prompt)
    return True


def _pending_tool_calls(items: Sequence[Item]) -> list[FunctionCall]:
    """Function calls in `items` that have no matching output yet."""
    answered = {i.call_id for i in items if isinstance(i, FunctionCallOutput)}
    return [
        i for i in items if isinstance(i, FunctionCall) and i.call_id not in answered
    ]


def run_react_loop(
    agent: "Agent",
    prompt: InputItem,
    options: LLMOptions,
) -> Run:
    tools = agent.tools

    async def gen() -> AsyncGenerator[Item, None]:
        if not await __add_prompt(agent, agent.history, prompt):
            return
        items = agent.history.get()
        instructions = agent.history.get_instructions()

        async with httpx.AsyncClient() as client:
            while True:
                result = await agent.provider.generate(
                    instructions=instructions,
                    items=items,
                    tools=tools,
                    options=options,
                    client=client,
                )
                c.usage += result.usage
                c.last_usage = result.usage
                for item in result.items:
                    c._add_new_item(item)
                    items.append(item)
                    yield item
                tool_calls = _pending_tool_calls(result.items)
                if result.finish_reason != "tool-calls" or not tool_calls:
                    break
                # Call tools and continue
                for output in await __process_tool_calls(
                    agent,
                    tool_calls,
                    tools=tools,
                    parallel=options.parallel_tool_calls or False,
                ):
                    c._add_new_item(output)
                    items.append(output)
                    yield output
                # Process enqueued items if any
                for m in agent.clear_deferred_items():
                    c._add_new_item(m)
                    items.append(m)

        agent.history.add(*c.new_items)

    async def gen_wrapper() -> AsyncGenerator[Item, None]:
        from agentia.tools.mcp import MCPContext

        await agent._auto_compact_if_needed(options)

        async with MCPContext() as _ctx:
            agent._temp_mcp_context = _ctx

            await tools.init()

            async for item in gen():
                yield item
            agent._temp_mcp_context = None

    c = Run(gen_wrapper(), agent)
    return c


def run_react_loop_streamed(
    agent: "Agent",
    prompt: InputItem,
    events: bool,
    options: LLMOptions,
) -> RunStream | RunEvents:
    tools = agent.tools

    async def gen() -> AsyncGenerator[StreamEvent, None]:
        if not await __add_prompt(agent, agent.history, prompt):
            return
        items = agent.history.get()
        last_finish_reason: FinishReason = "unknown"
        seq = 0

        def next_seq() -> int:
            nonlocal seq
            seq += 1
            return seq

        yield RunStartedEvent(sequence_number=next_seq())

        async with httpx.AsyncClient() as client:
            while True:
                turn_items: list[Item] = []
                async for event in agent.provider.stream(
                    instructions=instructions_of(agent),
                    items=items,
                    tools=tools,
                    options=options,
                    client=client,
                ):
                    event.sequence_number = next_seq()
                    match event.type:
                        case "response.output_item.done":
                            turn_items.append(event.item)
                        case "response.completed" | "response.incomplete":
                            s.usage += event.response.usage
                            s.last_usage = event.response.usage
                            last_finish_reason = event.response.finish_reason
                        case "response.failed":
                            s.usage += event.response.usage
                            s.last_usage = event.response.usage
                            last_finish_reason = "error"
                    yield event

                for item in turn_items:
                    items.append(item)
                    s._add_new_item(item)

                tool_calls = _pending_tool_calls(turn_items)
                if not tool_calls:
                    break

                # Tool results are inputs to the next request, not model output, so
                # they get the agentia extension event rather than output_item.done.
                for output in await __process_tool_calls(
                    agent, tool_calls, tools, options.parallel_tool_calls or False
                ):
                    items.append(output)
                    s._add_new_item(output)
                    yield InputItemAddedEvent(sequence_number=next_seq(), item=output)

                # Process enqueued items if any
                for m in agent.clear_deferred_items():
                    items.append(m)
                    s._add_new_item(m)
                    yield InputItemAddedEvent(sequence_number=next_seq(), item=m)

        s.finish_reason = last_finish_reason
        yield RunCompletedEvent(
            sequence_number=next_seq(),
            usage=s.usage,
            finish_reason=last_finish_reason,
        )

        agent.history.add(*s.new_items)

    async def gen_wrapper() -> AsyncGenerator[StreamEvent, None]:
        from agentia.tools.mcp import MCPContext

        await agent._auto_compact_if_needed(options)

        async with MCPContext() as _ctx:
            agent._temp_mcp_context = _ctx

            await tools.init()

            async for event in gen():
                yield event
            agent._temp_mcp_context = None

    if events:
        s = RunEvents(gen_wrapper(), agent)
    else:
        s = RunStream(gen_wrapper(), agent)
    return s


def instructions_of(agent: "Agent") -> str:
    return agent.history.get_instructions()
