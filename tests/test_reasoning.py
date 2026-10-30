from agentia import Agent, AssistantMessage, ReasoningPart
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


# Claude returns raw reasoning text with a signature; OpenAI returns a summary with
# encrypted reasoning.
MODELS = ["anthropic/claude-haiku-4.5:think", "openai/gpt-5-mini:think"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_reasoning(model: str):
    os.environ["OPENROUTER_REASONING_EFFORT"] = "high"

    agent = Agent(model=model)

    run = agent.run("Think for a while; Hi?")

    has_reasoning = False

    async for msg in run:
        print(msg)
        if msg.role == "assistant":
            r = msg.reasoning
            if r and r.strip() != "":
                has_reasoning = True
    assert has_reasoning


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_reasoning_stream(model: str):
    os.environ["OPENROUTER_REASONING_EFFORT"] = "high"

    agent = Agent(model=model)

    run = agent.run("Think for a while; Hi?", stream=True)

    has_reasoning = False

    async for msg in run:
        if isinstance(msg, ReasoningStream):
            reasoning = await msg
            print(reasoning)
            has_reasoning = True

    assert has_reasoning


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("stream", [False, True])
async def test_reasoning_is_replayed_with_tools(model: str, stream: bool):
    agent = Agent(model=model, tools=[get_current_weather])
    prompt = "What's the weather in Boston? Use the tool."
    if stream:
        run = agent.run(prompt, stream=True)
        async for _ in run:
            pass
        await run
    else:
        await agent.run(prompt)

    reasoning = [
        p
        for m in agent.history.get()
        if isinstance(m, AssistantMessage)
        for p in m.content
        if isinstance(p, ReasoningPart)
    ]
    assert reasoning
    # The provider's reasoning item is kept, so it can be replayed
    for r in reasoning:
        assert r.provider_data and r.provider_data.get("type") == "reasoning"
        assert r.provider_data.get("encrypted_content") or r.provider_data.get(
            "signature"
        )

    # The next turn sends the reasoning (and its signature) back
    response = await agent.run("And in Paris? Use the tool again.")
    assert response.text
