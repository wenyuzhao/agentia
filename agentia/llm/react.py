from typing import TYPE_CHECKING, AsyncGenerator, Optional, Sequence
import httpx
from agentia.history import History
from agentia.llm import LLMOptions
from agentia.llm.completion import ChatCompletion
from agentia.llm.stream import ChatCompletionEvents, ChatCompletionStream
from agentia.models import *
from agentia.tools.tools import ToolSet

if TYPE_CHECKING:
    from agentia.agent import Agent
    from agentia.live import LiveOptions


async def __process_tool_calls(
    agent: "Agent", tool_calls: list[ToolCall], tools: ToolSet, parallel: bool
) -> tuple[ToolMessage, list[ToolCallResponse]]:
    assert tools is not None, "No tools provided"
    tool_responses = await tools.run(agent, tool_calls, parallel)
    tool_msg_parts: list[MessagePartToolResult] = []
    for tr in tool_responses:
        mptr = MessagePartToolResult(
            tool_call_id=tr.tool_call_id,
            tool_name=tr.tool_name,
            input=tr.input,
            output=tr.output,
            output_files=tr.output_files,
        )
        tool_msg_parts.append(mptr)
    tool_msg = ToolMessage(content=tool_msg_parts)
    return tool_msg, tool_responses


def __add_prompt(
    _agent: "Agent", history: History, prompt: str | Message | Sequence[Message]
) -> None:
    if isinstance(prompt, str):
        history.add(UserMessage(content=[MessagePartText(text=prompt)]))
    elif not isinstance(prompt, (list, Sequence)):
        history.add(prompt)
    else:
        history.add(*prompt)


def run_react_loop(
    agent: "Agent",
    prompt: str | Message | Sequence[Message],
    options: LLMOptions,
    live_options: Optional["LiveOptions"],
) -> ChatCompletion:
    tools = agent.tools

    async def gen() -> AsyncGenerator[AssistantMessage | ToolMessage, None]:
        __add_prompt(agent, agent.history, prompt)
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
        from agentia.live import LiveOptions

        await agent._auto_compact_if_needed(options)

        async with MCPContext() as _ctx:
            agent._temp_mcp_context = _ctx

            await tools.init()

            if agent.provider.supports_live:
                await agent.provider.connect_live(
                    live_options or LiveOptions(),
                    tools,
                    agent.history.get_instructions(),
                    history=agent.history,
                )

            async for msg in gen():
                yield msg
            agent._temp_mcp_context = None

            if agent.provider.supports_live:
                await agent.provider.disconnect_live()

    c = ChatCompletion(gen_wrapper(), agent)
    return c


def run_react_loop_streamed(
    agent: "Agent",
    prompt: str | Message | Sequence[Message],
    events: bool,
    options: LLMOptions,
    live_options: Optional["LiveOptions"],
) -> ChatCompletionStream | ChatCompletionEvents:
    tools = agent.tools

    async def gen() -> AsyncGenerator[StreamPart, None]:
        __add_prompt(agent, agent.history, prompt)
        messages = agent.history.get()
        last_finish_reason: FinishReason = "unknown"
        started = False
        instructions = agent.history.get_instructions()

        async with httpx.AsyncClient() as client:
            while True:
                tool_calls: list[ToolCall] = []
                parts: list[AssistantMessagePart] = []
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
                            parts.append(MessagePartText(text=last_msg))
                        case "reasoning-start":
                            last_reasoning = ""
                        case "reasoning-delta":
                            last_reasoning += part.delta
                        case "reasoning-end":
                            parts.append(MessagePartReasoning(text=last_reasoning))
                        case "tool-call":
                            parts.append(
                                MessagePartToolCall(
                                    tool_call_id=part.tool_call_id,
                                    tool_name=part.tool_name,
                                    input=part.input,
                                    provider_executed=part.provider_executed,
                                )
                            )
                            if not part.provider_executed:
                                tool_calls.append(part)

                    if part.type == "turn-end":
                        s.usage += part.usage
                        s.last_usage = part.usage
                        last_finish_reason = part.finish_reason
                    else:
                        yield part
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
        from agentia.live import LiveOptions

        await agent._auto_compact_if_needed(options)

        async with MCPContext() as _ctx:
            agent._temp_mcp_context = _ctx

            await tools.init()

            if agent.provider.supports_live:
                await agent.provider.connect_live(
                    live_options or LiveOptions(),
                    tools,
                    agent.history.get_instructions(),
                    history=agent.history,
                )

            async for part in gen():
                yield part
            agent._temp_mcp_context = None

            if agent.provider.supports_live:
                await agent.provider.disconnect_live()

    if events:
        s = ChatCompletionEvents(gen_wrapper(), agent)
    else:
        s = ChatCompletionStream(gen_wrapper(), agent)
    return s


async def run_react_loop_live(agent: "Agent") -> AsyncGenerator[StreamPart, None]:
    started = False
    parts: list[AssistantMessagePart] = []
    tool_calls: list[ToolCall] = []
    last_msg = ""
    last_reasoning = ""

    async for part in agent.provider.receive():
        if not started:
            started = True
            yield StreamPartStart()
        assert part.type not in (
            "stream-start",
            "stream-end",
        ), f"Provider must not yield {part.type}"

        match part.type:
            case "turn-start":
                yield part
            case "text-start":
                last_msg = ""
            case "text-delta":
                last_msg += part.delta
            case "text-end":
                parts.append(MessagePartText(text=last_msg))
            case "reasoning-start":
                last_reasoning = ""
            case "reasoning-delta":
                last_reasoning += part.delta
            case "reasoning-end":
                parts.append(MessagePartReasoning(text=last_reasoning))
            case "tool-call":
                parts.append(
                    MessagePartToolCall(
                        tool_call_id=part.tool_call_id,
                        tool_name=part.tool_name,
                        input=part.input,
                        provider_executed=part.provider_executed,
                    )
                )
                if not part.provider_executed:
                    tool_calls.append(part)
        yield part

        if part.type == "turn-end":
            if part.usage.total_tokens is not None:
                agent.history.current_tokens = part.usage.total_tokens
            agent.history.usage += part.usage

        if part.type == "turn-end" or part.type == "tool-call":
            # append the assistant message to history
            msg = AssistantMessage(content=parts)
            agent.history.add(msg)
            # run tool calls if any
            if tool_calls:
                yield StreamPartTurnStart(role="tool")
                tool_msg, tool_responses = await __process_tool_calls(
                    agent, tool_calls, agent.tools, False
                )
                for tr in tool_responses:
                    yield tr
                agent.history.add(tool_msg)
                yield StreamPartTurnEnd(role="tool", message=tool_msg)
                await agent.provider.send_tool_responses(tool_responses)
            # reset for next turn
            parts = []
            tool_calls = []
            last_msg = ""
            last_reasoning = ""
