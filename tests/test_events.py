import json
from typing import Annotated

import dotenv
import pytest

from agentia import Agent, FunctionCall, FunctionCallOutput

dotenv.load_dotenv()

pytestmark = pytest.mark.live


def calc(expression: Annotated[str, "The expression to calculate"]):
    """Calculate the result of a mathematical expression"""
    return eval(expression)


@pytest.mark.asyncio
async def test_events():
    agent = Agent(model="openai/gpt-5-nano", tools=[calc])
    run = agent.run("Use tools to calculate 1 + 1", stream=True, events=True)
    call = None
    output = None
    types: list[str] = []
    async for e in run:
        print(e)
        types.append(e.type)
        if e.type == "response.output_item.done" and isinstance(e.item, FunctionCall):
            call = e.item
        elif e.type == "agentia.input_item.added" and isinstance(
            e.item, FunctionCallOutput
        ):
            output = e.item
    assert call is not None
    assert output is not None
    assert call.name == "calc"
    assert output.name == "calc"
    assert json.loads(output.text) == 2 or output.text == "2"
    # The run is wrapped in agentia lifecycle events, and each turn in a response.
    assert types[0] == "agentia.run.started"
    assert types[-1] == "agentia.run.completed"
    assert "response.created" in types
    assert "response.completed" in types


@pytest.mark.asyncio
async def test_event_sequence_numbers_are_monotonic():
    agent = Agent(model="openai/gpt-5-nano", tools=[calc])
    run = agent.run("Use tools to calculate 2 + 3", stream=True, events=True)
    last = -1
    async for e in run:
        assert e.sequence_number > last
        last = e.sequence_number
    assert last > 0
