from agentia import Agent
import pytest
import dotenv

dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_compact_basic():
    agent = Agent(
        model="openai/gpt-5-mini", instructions="You are a helpful assistant."
    )
    # Simulate a multi-turn conversation
    await agent.run("My name is Alice and I live in Paris.")
    await agent.run("I work as a software engineer at Acme Corp.")
    await agent.run("What is 2 + 2?")

    # Compact the history
    await agent.compact(effort="medium")

    # Only the last message is kept; the rest is replaced by the summary
    messages = agent.history.get()
    assert len(messages) == 1
    assert len(agent.history.get_all()) == 6
    # The summary should contain key facts, and is part of the instructions
    assert agent.history.summary is not None
    assert "alice" in agent.history.summary.lower()
    assert agent.history.summary in agent.history.get_instructions()

    # The conversation continues with the summary as context
    response = await agent.run("What is my name?")
    assert "alice" in response.text.lower()


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

    assert len(agent.history.get()) == 1
    assert agent.history.summary is not None
    assert "12345" in agent.history.summary
