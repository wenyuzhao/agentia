from agentia import Agent, ToolCall, ToolResult
from typing import Annotated
import pytest
import dotenv

dotenv.load_dotenv()


def calc(expression: Annotated[str, "The expression to calculate"]):
    """Calculate the result of a mathematical expression"""
    return eval(expression)


@pytest.mark.asyncio
async def test_events():
    agent = Agent(model="openai/gpt-5-nano", tools=[calc])
    run = agent.run("Use tools to calculate 1 + 1", stream=True, events=True)
    tool_call_part = None
    tool_result_part = None
    async for e in run:
        print(e)
        if isinstance(e, ToolCall):
            tool_call_part = e
        elif isinstance(e, ToolResult):
            tool_result_part = e
    assert tool_call_part is not None
    assert tool_result_part is not None
    assert tool_call_part.tool_name == "calc"
    assert tool_result_part.tool_name == "calc"
    assert tool_result_part.result == 2
