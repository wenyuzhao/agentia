from agentia import Agent
from agentia.models import CompactedMessage, SystemMessage
import pytest
import dotenv

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

    messages = agent.history.get(include_instructions=True)
    # Should have system instructions + compacted message + last message
    assert len(messages) == 3
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], CompactedMessage)
    # The compacted message should contain key facts
    text = messages[1].content.lower()
    assert "alice" in text


@pytest.mark.asyncio
async def test_compact_empty_history():
    agent = Agent(
        model="openai/gpt-5-nano", instructions="You are a helpful assistant."
    )
    # Compacting empty history should be a no-op
    await agent.compact(effort="low")
    messages = agent.history.get(include_instructions=False)
    assert len(messages) == 0


@pytest.mark.asyncio
async def test_compact_preserves_instructions():
    instructions = "You are a pirate assistant who speaks in pirate slang."
    agent = Agent(model="openai/gpt-5-nano", instructions=instructions)
    await agent.run("Hello there!")

    await agent.compact(effort="high")

    messages = agent.history.get(include_instructions=True)
    assert isinstance(messages[0], SystemMessage)
    assert "pirate" in messages[0].content


@pytest.mark.asyncio
async def test_compact_with_model_override():
    agent = Agent(
        model="openai/gpt-5-nano", instructions="You are a helpful assistant."
    )
    await agent.run("Remember that the secret code is 12345.")

    # Use a different model for compaction
    await agent.compact(effort="medium", model="openai/gpt-5-nano")

    messages = agent.history.get(include_instructions=False)
    assert len(messages) == 2
    assert isinstance(messages[0], CompactedMessage)
    assert "12345" in messages[0].content
