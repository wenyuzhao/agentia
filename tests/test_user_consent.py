from agentia import Agent, UserConsentEvent, ToolResult
from typing import Annotated
import pytest
import dotenv

dotenv.load_dotenv()


user_consent_should_pass = True


def calc(expression: Annotated[str, "The expression to calculate"]):
    """Calculate the result of a mathematical expression"""
    if not (yield UserConsentEvent(message="Are you sure you want to calculate this?")):
        assert not user_consent_should_pass
        return "User did not consent to calculation."
    assert user_consent_should_pass
    return eval(expression)


async def calc2(expression: Annotated[str, "The expression to calculate"]):
    """Calculate the result of a mathematical expression"""
    if not (yield UserConsentEvent(message="Are you sure you want to calculate this?")):
        assert not user_consent_should_pass
        raise ToolResult("User did not consent to calculation.")
    assert user_consent_should_pass
    raise ToolResult(eval(expression))


@pytest.mark.asyncio
async def test_user_consent_pass():
    agent = Agent(model="openai/gpt-4.1-nano", tools=[calc])
    run = agent.run("Calculate 1 + 1", events=True)
    async for msg in run:
        if isinstance(msg, UserConsentEvent):
            assert msg.message == "Are you sure you want to calculate this?"
            global user_consent_should_pass
            user_consent_should_pass = True
            msg.response = True


@pytest.mark.asyncio
async def test_user_consent_fail():
    agent = Agent(model="openai/gpt-4.1-nano", tools=[calc])
    run = agent.run("Calculate 1 + 1", events=True)
    async for msg in run:
        if isinstance(msg, UserConsentEvent):
            assert msg.message == "Are you sure you want to calculate this?"
            global user_consent_should_pass
            user_consent_should_pass = False
            msg.response = False


@pytest.mark.asyncio
async def test_user_consent_pass_async():
    agent = Agent(model="openai/gpt-4.1-nano", tools=[calc2])
    run = agent.run("Calculate 1 + 1", events=True)
    async for msg in run:
        if isinstance(msg, UserConsentEvent):
            assert msg.message == "Are you sure you want to calculate this?"
            global user_consent_should_pass
            user_consent_should_pass = True
            msg.response = True


@pytest.mark.asyncio
async def test_user_consent_fail_async():
    agent = Agent(model="openai/gpt-4.1-nano", tools=[calc2])
    run = agent.run("Calculate 1 + 1", events=True)
    async for msg in run:
        if isinstance(msg, UserConsentEvent):
            assert msg.message == "Are you sure you want to calculate this?"
            global user_consent_should_pass
            user_consent_should_pass = False
            msg.response = False
