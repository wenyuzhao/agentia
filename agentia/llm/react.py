from typing import TYPE_CHECKING, AsyncGenerator
import httpx
from agentia.history import History
from agentia.llm import LLMOptions
from agentia.llm.completion import ChatCompletion
from agentia.llm.stream import ChatCompletionEvents, ChatCompletionStream
from agentia.models import *
from agentia.tools.tools import ToolSet

if TYPE_CHECKING:
    from agentia.agent import Agent


async def __process_tool_calls(
    agent: "Agent", tool_calls: list[ToolCallPart], tools: ToolSet, parallel: bool
) -> tuple[ToolMessage, list[ToolResultPart]]:
    assert tools is not None, "No tools provided"
    tool_responses = await tools.run(agent, tool_calls, parallel)
    return ToolMessage(content=tool_responses), tool_responses


async def __add_prompt(agent: "Agent", history: History, prompt: Prompt) -> bool:
    """Add the prompt to history. Returns False if the prompt was a slash command
    that produced no output, in which case the LLM should be skipped."""

    prompt = await agent.commands.process_messages(prompt)

    if not prompt:
        return False

    history.add(*prompt)
    return True


def run_react_loop(
    agent: "Agent",
    prompt: Prompt,
    options: LLMOptions,
) -> ChatCompletion:
    tools = agent.tools

    async def gen() -> AsyncGenerator[AssistantMessage | ToolMessage, None]:
        if not await __add_prompt(agent, agent.history, prompt):
            return
        messages = agent.history.get()
        instructions = agent.history.get_instructions()

        async with httpx.AsyncClient() as client:
            while True:
                result = await agent.provider.generate(
                    instructions=instructions,
                    messages=messages,
                    tools=tools,
                    options=options,
                    client=client,
                )
                c.usage += result.usage
                c.last_usage = result.usage
                c._add_new_message(result.message)
                messages.append(result.message)
                yield result.message
                # Add new messages
                tool_calls = result.message.tool_calls
                if result.finish_reason != "tool-calls":
                    break
                # Call tools and continue
                tool_msg, _ = await __process_tool_calls(
                    agent,
                    tool_calls,
                    tools=tools,
                    parallel=options.parallel_tool_calls or False,
                )
                c._add_new_message(tool_msg)
                messages.append(tool_msg)
                yield tool_msg
                # Process enqueued messages if any
                if msgs := agent.clear_deferred_messages():
                    for m in msgs:
                        c._add_new_message(m)
                        messages.append(m)

        agent.history.add(*c.new_messages)

    async def gen_wrapper() -> AsyncGenerator[AssistantMessage | ToolMessage, None]:
        from agentia.tools.mcp import MCPContext

        await agent._auto_compact_if_needed(options)

        async with MCPContext() as _ctx:
            agent._temp_mcp_context = _ctx

            await tools.init()

            async for msg in gen():
                yield msg
            agent._temp_mcp_context = None

    c = ChatCompletion(gen_wrapper(), agent)
    return c


def run_react_loop_streamed(
    agent: "Agent",
    prompt: Prompt,
    events: bool,
    options: LLMOptions,
) -> ChatCompletionStream | ChatCompletionEvents:
    tools = agent.tools

    async def gen() -> AsyncGenerator[StreamPart, None]:
        if not await __add_prompt(agent, agent.history, prompt):
            return
        messages = agent.history.get()
        last_finish_reason: FinishReason = "unknown"
        started = False
        instructions = agent.history.get_instructions()

        async with httpx.AsyncClient() as client:
            while True:
                tool_calls: list[ToolCallPart] = []
                parts: list[AssistantMessagePart] = []
                annotations: list[UrlCitation] = []
                last_msg = ""
                last_reasoning = ""
                async for part in agent.provider.stream(
                    instructions=instructions,
                    messages=messages,
                    tools=tools,
                    options=options,
                    client=client,
                ):
                    if not started:
                        started = True
                        yield StreamPartStart()

                    assert part.type not in (
                        "stream-start",
                        "stream-end",
                    ), f"Provider must not yield {part.type}"

                    match part.type:
                        case "text-start":
                            last_msg = ""
                        case "text-delta":
                            last_msg += part.delta
                        case "text-end":
                            parts.append(TextPart(last_msg))
                        case "reasoning-start":
                            last_reasoning = ""
                        case "reasoning-delta":
                            last_reasoning += part.delta
                        case "reasoning-end":
                            parts.append(
                                ReasoningPart(
                                    last_reasoning, provider_data=part.provider_data
                                )
                            )
                        case "tool-call":
                            parts.append(part)
                            tool_calls.append(part)
                        case "file":
                            parts.append(part)
                        case "url-citation":
                            annotations.append(part)

                    if part.type == "turn-end":
                        s.usage += part.usage
                        s.last_usage = part.usage
                        last_finish_reason = part.finish_reason
                    else:
                        yield part
                if annotations:
                    # The provider reports citations for the whole message, which is
                    # normally a single text part.
                    text_parts = [p for p in parts if isinstance(p, TextPart)]
                    if text_parts:
                        text_parts[-1].annotations = annotations
                msg = AssistantMessage(content=parts)
                yield StreamPartTurnEnd(
                    role="assistant",
                    message=msg,
                    usage=s.usage,
                    finish_reason=last_finish_reason,
                )
                messages.append(msg)
                s._add_new_message(msg)
                if not tool_calls:
                    break
                # Call tools and continue
                yield StreamPartTurnStart(role="tool")
                tool_msg, tool_responses = await __process_tool_calls(
                    agent,
                    tool_calls,
                    tools,
                    options.parallel_tool_calls or False,
                )
                for tr in tool_responses:
                    yield tr
                yield StreamPartTurnEnd(role="tool", message=tool_msg)
                messages.append(tool_msg)
                s._add_new_message(tool_msg)
                # Process enqueued messages if any
                if msgs := agent.clear_deferred_messages():
                    for m in msgs:
                        s._add_new_message(m)
                        messages.append(m)
        s.finish_reason = last_finish_reason
        yield StreamPartEnd(usage=s.usage, finish_reason=last_finish_reason)

        agent.history.add(*s.new_messages)

    async def gen_wrapper() -> AsyncGenerator[StreamPart, None]:
        from agentia.tools.mcp import MCPContext

        await agent._auto_compact_if_needed(options)

        async with MCPContext() as _ctx:
            agent._temp_mcp_context = _ctx

            await tools.init()

            async for part in gen():
                yield part
            agent._temp_mcp_context = None

    if events:
        s = ChatCompletionEvents(gen_wrapper(), agent)
    else:
        s = ChatCompletionStream(gen_wrapper(), agent)
    return s
