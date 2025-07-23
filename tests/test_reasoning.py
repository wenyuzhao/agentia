from agentia import Agent
from typing import Literal, Annotated
import pytest
import dotenv
import os

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

    agent = Agent(model="deepseek/deepseek-r1")

    run = agent.run("Hi?")

    has_reasoning = False
    has_content = False

    async for msg in run:
        print(msg.reasoning)
        assert (msg.reasoning or "").strip() != ""
        has_reasoning = True
        print(msg.content)
        assert (msg.content or "").strip() != ""
        has_content = True

    assert has_reasoning and has_content


@pytest.mark.asyncio
async def test_reasoning_stream():
    os.environ["OPENROUTER_REASONING_EFFORT"] = "high"

    agent = Agent(model="deepseek/deepseek-r1")

    run = agent.run("Hi?", stream=True)

    has_reasoning = False
    has_content = False

    async for msg in run:
        assert msg.reasoning
        reasoning = await msg.reasoning
        print(reasoning)
        assert reasoning.strip() != ""
        has_reasoning = True
        content = await msg
        print(content)
        assert (content.content or "").strip() != ""
        has_content = True

    assert has_reasoning and has_content
