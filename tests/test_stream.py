from agentia import Agent
from typing import Literal, Annotated
import pytest
import dotenv

from agentia.llm.stream import TextStream
from agentia.llm.stream import TextStream

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
async def test_stream():
    agent = Agent(model="openai/gpt-4.1-nano", tools=[get_current_weather])
    run = agent.run("What is the weather like in boston?", stream=True)
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
    agent = Agent(model="openai/gpt-4.1-nano", tools=[get_current_weather])
    run = agent.run("What is the weather like in boston?", stream=True, events=True)
    all_assistant_content = ""
    async for stream in run:
        print("stream: ", stream)
        content = ""
        if stream.type == "text-delta":
            content += stream.delta
        all_assistant_content += content
    print(all_assistant_content)
    assert "72" in all_assistant_content
