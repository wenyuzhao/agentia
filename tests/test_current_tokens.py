from agentia import Agent, Usage
from agentia.llm.stream import TextStream
from typing import Annotated
import pytest
import dotenv

dotenv.load_dotenv()


def add(a: Annotated[int, "first number"], b: Annotated[int, "second number"]):
    """Add two numbers"""
    return a + b


# --- current_tokens / current_context_length ---


@pytest.mark.asyncio
async def test_current_tokens_after_run():
    agent = Agent(model="openai/gpt-5-nano")
    assert agent.current_context_length == 0
    await agent.run("Say hello")
    assert agent.current_context_length > 0


@pytest.mark.asyncio
async def test_current_tokens_after_tool_call():
    agent = Agent(model="openai/gpt-5-nano", tools=[add])
    await agent.run("Use the add tool to compute 2 + 3")
    assert agent.current_context_length > 0


@pytest.mark.asyncio
async def test_current_tokens_increases_over_turns():
    agent = Agent(model="openai/gpt-5-nano")
    await agent.run("Say hello")
    tokens_after_first = agent.current_context_length
    assert tokens_after_first > 0

    await agent.run("Say goodbye")
    tokens_after_second = agent.current_context_length
    assert tokens_after_second > tokens_after_first


@pytest.mark.asyncio
async def test_current_tokens_reset_on_clear():
    agent = Agent(model="openai/gpt-5-nano")
    await agent.run("Say hello")
    assert agent.current_context_length > 0
    agent.history.clear()
    assert agent.current_context_length == 0


# --- usage accumulation ---


@pytest.mark.asyncio
async def test_usage_after_run():
    agent = Agent(model="openai/gpt-5-nano")
    assert agent.usage == Usage()
    await agent.run("Say hello")
    assert agent.usage.total_tokens is not None
    assert agent.usage.total_tokens > 0
    assert agent.usage.input_tokens is not None
    assert agent.usage.output_tokens is not None


@pytest.mark.asyncio
async def test_usage_accumulates_over_turns():
    agent = Agent(model="openai/gpt-5-nano")
    await agent.run("Say hello")
    usage_after_first = agent.usage.total_tokens
    assert usage_after_first is not None and usage_after_first > 0

    await agent.run("Say goodbye")
    usage_after_second = agent.usage.total_tokens
    assert usage_after_second is not None
    assert usage_after_second > usage_after_first


@pytest.mark.asyncio
async def test_usage_with_tool_calls():
    agent = Agent(model="openai/gpt-5-nano", tools=[add])
    await agent.run("Use the add tool to compute 2 + 3")
    assert agent.usage.total_tokens is not None
    assert agent.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_usage_reset_on_clear():
    agent = Agent(model="openai/gpt-5-nano")
    await agent.run("Say hello")
    assert agent.usage.total_tokens is not None
    agent.history.clear()
    assert agent.usage == Usage()


# --- streaming ---


@pytest.mark.asyncio
async def test_current_tokens_after_stream():
    agent = Agent(model="openai/gpt-5-nano")
    run = agent.run("Say hello", stream=True)
    async for item in run:
        if isinstance(item, TextStream):
            await item
    assert agent.current_context_length > 0


@pytest.mark.asyncio
async def test_current_tokens_after_stream_events():
    agent = Agent(model="openai/gpt-5-nano")
    run = agent.run("Say hello", stream=True, events=True)
    async for _ in run:
        pass
    assert agent.current_context_length > 0
