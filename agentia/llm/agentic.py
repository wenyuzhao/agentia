from typing import TYPE_CHECKING, AsyncGenerator
import httpx
from agentia.llm import LLMOptions
from agentia.llm.completion import ChatCompletion
from agentia.llm.stream import ChatCompletionEvents, ChatCompletionStream
from agentia.spec import *
from agentia.tools.tools import ToolSet

if TYPE_CHECKING:
    from agentia.agent import Agent


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


def run_agent_loop(agent: "Agent", options: LLMOptions) -> ChatCompletion:
    tools = agent.tools

    async def gen() -> AsyncGenerator[AssistantMessage | ToolMessage, None]:
        await tools.init()
        messages = agent.history.get()
        async with httpx.AsyncClient() as client:
            while True:
                result = await agent.provider.generate(
                    messages=messages, tools=tools, options=options, client=client
                )
                c.usage += result.usage
                c._add_new_message(result.message)
                yield result.message
                # Add new messages
                messages.append(result.message)
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
                yield tool_msg
                messages.append(tool_msg)
                c._add_new_message(tool_msg)

        agent.history.add(*c.new_messages)

    async def gen_with_mcp_context() -> (
        AsyncGenerator[AssistantMessage | ToolMessage, None]
    ):
        from agentia.tools.mcp import MCPContext

        async with MCPContext() as _ctx:
            agent._temp_mcp_context = _ctx
            async for msg in gen():
                yield msg
            agent._temp_mcp_context = None

    c = ChatCompletion(gen_with_mcp_context(), agent)
    return c


def run_agent_loop_streamed(
    agent: "Agent", events: bool, options: LLMOptions
) -> ChatCompletionStream | ChatCompletionEvents:
    tools = agent.tools

    async def gen() -> AsyncGenerator[StreamPart, None]:
        await tools.init()
        messages = agent.history.get()
        last_finish_reason: FinishReason = "unknown"

        async with httpx.AsyncClient() as client:
            while True:
                tool_calls: list[ToolCall] = []
                parts: list[AssistantMessagePart] = []
                last_msg = ""
                last_reasoning = ""
                async for part in agent.provider.stream(
                    messages, tools, options, client
                ):

                    match part.type:
                        case "stream-start":
                            ...
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

                    if part.type == "finish":
                        s.usage += part.usage
                        last_finish_reason = part.finish_reason
                    else:
                        yield part
                msg = AssistantMessage(content=parts)
                messages.append(msg)
                s._add_new_message(msg)
                if not tool_calls:
                    break
                # Call tools and continue
                tool_msg, tool_responses = await __process_tool_calls(
                    agent,
                    tool_calls,
                    tools,
                    options.parallel_tool_calls or False,
                )
                for tr in tool_responses:
                    yield tr
                messages.append(tool_msg)
                s._add_new_message(tool_msg)
        s.finish_reason = last_finish_reason
        yield StreamPartStreamEnd(usage=s.usage, finish_reason=last_finish_reason)

        agent.history.add(*s.new_messages)

    async def gen_with_mcp_context() -> AsyncGenerator[StreamPart, None]:
        from agentia.tools.mcp import MCPContext

        async with MCPContext() as _ctx:
            agent._temp_mcp_context = _ctx
            async for part in gen():
                yield part
            agent._temp_mcp_context = None

    if events:
        s = ChatCompletionEvents(gen_with_mcp_context(), agent)
    else:
        s = ChatCompletionStream(gen_with_mcp_context(), agent)
    return s
