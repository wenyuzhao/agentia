from agentia import Agent, tool
from agentia.plugins import Plugin
from typing import Literal, Annotated
import pytest
import dotenv

dotenv.load_dotenv()


class FakeWeatherPlugin(Plugin):
    @tool
    def get_current_weather(
        self,
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


class FakeWeatherPlugin2(Plugin):
    @tool
    async def get_current_weather2(
        self,
        location: Annotated[str, "The city and state, e.g. San Francisco, CA"],
        unit: Literal["celsius", "fahrenheit"] | None = "fahrenheit",
    ):
        print("CALL get_current_weather2")
        """Get the current weather in a given location"""
        return {
            "location": location,
            "temperature": "72",
            "unit": unit,
            "forecast": ["sunny", "windy"],
        }


@pytest.mark.asyncio
async def test_plugin():
    agent = Agent(model="openai/gpt-4.1-nano", tools=[FakeWeatherPlugin2()])
    run = agent.run("What is the weather like in boston?")
    all_assistant_content: str = ""
    async for msg in run:
        if msg.role == "assistant":
            for p in msg.content:
                if p.type == "text":
                    all_assistant_content += p.text or ""
        print(msg)
    assert "72" in all_assistant_content
