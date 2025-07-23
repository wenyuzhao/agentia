from agentia import Agent, UserConsentEvent, ToolCallEvent, Event, tool
from typing import Annotated
import pytest
import dotenv

from agentia.message import is_event

dotenv.load_dotenv()

tool_received_metadata: str | None = None


@tool(metadata={"foo": "FOO"})
def calc(expression: Annotated[str, "The expression to calculate"]):
    """Calculate the result of a mathematical expression"""
    consent = UserConsentEvent(
        message="Are you sure you want to calculate this?", metadata={"bar": "BAR"}
    )
    confirmed = yield consent

    global tool_received_metadata
    tool_received_metadata = (consent.metadata or {}).get("baz")

    if not confirmed:
        return "User did not consent to calculation."

    return eval(expression)


@pytest.mark.asyncio
async def test_metadata():
    agent = Agent(model="openai/gpt-4.1-nano", tools=[calc])
    run = agent.run("Calculate 1 + 1", events=True)
    async for e in run:
        if is_event(e):
            print(e)
            if isinstance(e, ToolCallEvent):
                assert (e.metadata or {}).get("foo") == "FOO"
            if isinstance(e, UserConsentEvent):
                e.response = True
                assert e.metadata is not None
                assert e.metadata.get("bar") == "BAR"
                e.metadata["baz"] = "BAZ"
    global tool_received_metadata
    assert tool_received_metadata == "BAZ"
