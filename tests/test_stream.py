from agentia import Agent, Event
from typing import Literal, Annotated
import pytest
import dotenv

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
    agent = Agent(model="openai/gpt-4o-mini", tools=[get_current_weather])
    run = agent.run("What is the weather like in boston?", stream=True)
    all_assistant_content = ""
    async for stream in run:
        print("stream: ", stream)
        content = ""
        async for delta in stream:
            assert delta is None or isinstance(delta, str)
            content += delta
            print(" - ", delta)
        msg = await stream
        if msg.role == "assistant":
            all_assistant_content += content
        print(msg)
    assert "72" in all_assistant_content


@pytest.mark.asyncio
async def test_stream_with_events():
    agent = Agent(model="openai/gpt-4o-mini", tools=[get_current_weather])
    run = agent.run("What is the weather like in boston?", stream=True, events=True)
    all_assistant_content = ""
    async for stream in run:
        if isinstance(stream, Event):
            print(stream)
            continue
        print("stream: ", stream)
        content = ""
        async for delta in stream:
            assert delta is None or isinstance(delta, str)
            content += delta
            print(" - ", delta)
        msg = await stream
        if msg.role == "assistant":
            all_assistant_content += content
        print(msg)
    assert "72" in all_assistant_content
