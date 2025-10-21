from agentia import Agent
from typing import Literal, Annotated
import pytest
import dotenv
import os

from agentia.llm.stream import ReasoningStream

dotenv.load_dotenv()


def get_current_weather(
    location: Annotated[str, "The city and state, e.g. San Francisco, CA"],
    unit: Literal["celsius", "fahrenheit"] | None = "fahrenheit",
):
    """Get the current weather in a given location"""
    return {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }


@pytest.mark.asyncio
async def test_reasoning():
    os.environ["OPENROUTER_REASONING_EFFORT"] = "high"

    # agent = Agent(model="deepseek/deepseek-r1")
    agent = Agent(model="openai/o4-mini")

    run = agent.run("Hi?")

    has_reasoning = False

    async for msg in run:
        print(msg)
        for c in msg.content:
            if c.type == "reasoning":
                print("REASONING CONTENT:", c)
                assert (c.text or "").strip() != ""
                has_reasoning = True
    assert has_reasoning


@pytest.mark.asyncio
async def test_reasoning_stream():
    os.environ["OPENROUTER_REASONING_EFFORT"] = "high"

    agent = Agent(model="deepseek/deepseek-r1")

    run = agent.run("Hi?", stream=True)

    has_reasoning = False

    async for msg in run:
        if isinstance(msg, ReasoningStream):
            reasoning = await msg
            print(reasoning)
            has_reasoning = True

    assert has_reasoning
