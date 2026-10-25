import dotenv
import pytest

from agentia import Agent

dotenv.load_dotenv()

pytestmark = pytest.mark.live


async def test_compact_basic():
    agent = Agent(
        model="openai/gpt-5-nano", instructions="You are a helpful assistant."
    )
    await agent.run("My name is Alice and I live in Paris.")
    await agent.run("I work as a software engineer at Acme Corp.")
    await agent.run("What is 2 + 2?")

    c = await agent.compact(effort="medium", keep_exchanges=1)
    assert c is not None and c.covers == 4
    assert agent.history.compaction is c
    # The history keeps everything; the context is the last exchange.
    assert len(agent.history.items) == 6 and len(agent.history.context()) == 2
    assert "alice" in c.summary.lower()
    # The summary reaches the model.
    msg = await agent.run("What is my name?")
    assert "alice" in msg.text.lower()


async def test_compact_empty_history():
    agent = Agent(model="openai/gpt-5-nano")
    assert await agent.compact(effort="low") is None
    assert agent.history.items == []


async def test_compact_with_model_override():
    agent = Agent(model="openai/gpt-5-nano")
    await agent.run("Remember the code word: pineapple.")
    c = await agent.compact(keep_exchanges=0, model="openai/gpt-5-nano")
    assert c is not None and "pineapple" in c.summary.lower()
