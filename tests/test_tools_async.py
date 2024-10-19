from agentia import Agent, Message, tool
from typing import Literal,  Annotated
import pytest
import asyncio
import dotenv

dotenv.load_dotenv()


@tool
async def get_current_weather(
    location: Annotated[str, "The city and state, e.g. San Francisco, CA"],
    unit: Literal["celsius", "fahrenheit"] | None = "fahrenheit",
):
    """Get the current weather in a given location"""
    await asyncio.sleep(0.1)
    return {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }


@pytest.mark.asyncio
async def test_function_call():
    gpt = Agent(model="openai/gpt-4o-mini", tools=[get_current_weather])
    response = gpt.chat_completion(
        [Message(role="user", content="What is the weather like in boston?")]
    )
    all_assistant_content = ""
    async for msg in response.messages():
        if msg.role == "assistant":
            assert msg.content is None or isinstance(msg.content, str)
            all_assistant_content += msg.content or ""
        print(msg)
    assert "72" in all_assistant_content
