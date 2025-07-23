from agentia import Agent, UserConsentEvent, ToolCallEvent, Event
from typing import Annotated
import pytest
import dotenv
import copy

from agentia.message import is_event

dotenv.load_dotenv()


def calc(expression: Annotated[str, "The expression to calculate"]):
    """Calculate the result of a mathematical expression"""
    if not (yield UserConsentEvent(message="Are you sure you want to calculate this?")):
        return "User did not consent to calculation."
    return eval(expression)


@pytest.mark.asyncio
async def test_events():
    agent = Agent(model="openai/gpt-4.1-nano", tools=[calc])
    run = agent.run("Calculate 1 + 1", events=True)
    events: list[Event] = []
    async for e in run:
        if is_event(e):
            events.append(copy.copy(e))
            if isinstance(e, UserConsentEvent):
                e.response = True
    assert len(events) > 0
    assert (
        isinstance(events[0], ToolCallEvent)
        and events[0].name == "calc"
        and events[0].result is None
    )
    assert (
        isinstance(events[1], UserConsentEvent)
        and events[1].message == "Are you sure you want to calculate this?"
        and events[1].response is None
    )
    assert (
        isinstance(events[2], ToolCallEvent)
        and events[2].name == "calc"
        and events[2].result == 2
    )
