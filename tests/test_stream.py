from typing import Annotated, Literal

import dotenv
import pytest

from agentia import Agent
from agentia.llm.stream import TextStream

dotenv.load_dotenv()

pytestmark = pytest.mark.live


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
async def test_stream():
    agent = Agent(model="openai/gpt-5-nano", tools=[get_current_weather])
    run = agent.run(
        "What is the weather like in boston? Use tool and answer the question.",
        stream=True,
    )
    all_assistant_content = ""
    async for stream in run:
        print("stream: ", stream)
        if isinstance(stream, TextStream):
            msg = await stream
            print(msg)
            all_assistant_content += msg
    assert "72" in all_assistant_content


@pytest.mark.asyncio
async def test_stream_with_events():
    agent = Agent(model="openai/gpt-5-nano", tools=[get_current_weather])
    run = agent.run(
        "What is the weather like in boston? Use tool and answer the question.",
        stream=True,
        events=True,
    )
    all_assistant_content = ""
    async for event in run:
        print("stream: ", event)
        if event.type == "response.output_text.delta":
            all_assistant_content += event.delta
    print(all_assistant_content)
    assert "72" in all_assistant_content
