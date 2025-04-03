from agentia import Agent, UserMessage, tool, Plugin
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


@pytest.mark.asyncio
async def test_weather_and_memory_plugin():
    file = "_test-data.csv"
    with open(file, "w+") as f:
        f.write("")
    gpt = Agent(model="openai/gpt-4o-mini", tools=[FakeWeatherPlugin()])
    response = gpt.chat_completion(
        [UserMessage(content="What is the weather like in boston?")]
    )
    all_assistant_content: str = ""
    async for msg in response:
        if msg.role == "assistant":
            assert msg.content is None or isinstance(msg.content, str)
            all_assistant_content += msg.content or ""
        print(msg)
    assert "72" in all_assistant_content
