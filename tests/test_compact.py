import dotenv
import pytest

from agentia import Agent
from agentia.models import Compaction

dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_compact_basic():
    agent = Agent(
        model="openai/gpt-5-nano", instructions="You are a helpful assistant."
    )
    # Simulate a multi-turn conversation
    await agent.run("My name is Alice and I live in Paris.")
    await agent.run("I work as a software engineer at Acme Corp.")
    await agent.run("What is 2 + 2?")

    # Compact the history
    await agent.compact(effort="medium")

    messages = agent.history.get()
    # Should have the compaction item + the items kept from the last turn
    assert len(messages) == 2
    assert isinstance(messages[0], Compaction)
    # The compacted message should contain key facts
    text = messages[0].summary.lower()
    assert "alice" in text


@pytest.mark.asyncio
async def test_compact_empty_history():
    agent = Agent(
        model="openai/gpt-5-nano", instructions="You are a helpful assistant."
    )
    # Compacting empty history should be a no-op
    await agent.compact(effort="low")
    messages = agent.history.get()
    assert len(messages) == 0


@pytest.mark.asyncio
async def test_compact_with_model_override():
    agent = Agent(
        model="openai/gpt-5-nano", instructions="You are a helpful assistant."
    )
    await agent.run("Remember that the secret code is 12345.")

    # Use a different model for compaction
    await agent.compact(effort="medium", model="openai/gpt-5-nano")

    messages = agent.history.get()
    assert len(messages) == 2
    assert isinstance(messages[0], Compaction)
    assert "12345" in messages[0].summary
