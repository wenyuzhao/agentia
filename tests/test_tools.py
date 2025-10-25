from enum import StrEnum
from pydantic import BaseModel
from agentia import Agent
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
async def test_function_call():
    agent = Agent(model="openai/gpt-5-nano", tools=[get_current_weather])
    run = agent.run("What is the weather like in boston?")
    all_assistant_content = ""
    async for msg in run:
        if msg.role == "assistant":
            for p in msg.content:
                if p.type == "text":
                    all_assistant_content += p.text or ""
        print(msg)
    assert "72" in all_assistant_content


class Animal(StrEnum):
    """
    Must be either 'cat', 'dog', or 'bird'
    """

    cat = "cat"
    dog = "dog"
    bird = "bird"


@pytest.mark.asyncio
async def test_list_of_models_arg():
    status = {"called": False, "animals": []}

    async def feed_animals(
        animals: Annotated[
            list[Animal], "List of animals to feed. Must be greater than 0."
        ],
    ):
        """Feed a list of animals."""
        print(animals)
        if len(animals) == 0:
            raise BaseException("Animals list cannot be empty", animals)
        for a in animals:
            if not isinstance(a, Animal):
                raise BaseException("Invalid animal", a)
        status["called"] = True
        status["animals"] = animals
        return {
            "status": "success",
            "fed_animals": [a.value for a in animals],
        }

    agent = Agent(model="openai/gpt-5-nano", tools=[feed_animals])
    run = agent.run("Feed a cat and a bird.")
    async for msg in run:
        print(msg)
    assert status["called"] is True
    assert len(status["animals"]) == 2
    assert Animal.cat in status["animals"]
    assert Animal.bird in status["animals"]


class Args(BaseModel):
    animals: Annotated[list[Animal], "List of animals to feed. Must be greater than 0."]


@pytest.mark.asyncio
async def test_complex_arg():
    status = {"called": False, "animals": []}

    async def feed_animals(args: Args):
        """Feed a list of animals."""
        print(args.animals)
        if len(args.animals) == 0:
            raise BaseException("Animals list cannot be empty", args.animals)
        for a in args.animals:
            if not isinstance(a, Animal):
                raise BaseException("Invalid animal", a)
        status["called"] = True
        status["animals"] = args.animals
        return {
            "status": "success",
            "fed_animals": [a.value for a in args.animals],
        }

    agent = Agent(model="openai/gpt-5-nano", tools=[feed_animals])
    run = agent.run("Feed a cat and a bird.")
    async for msg in run:
        print(msg)
    assert status["called"] is True
    assert len(status["animals"]) == 2
    assert Animal.cat in status["animals"]
    assert Animal.bird in status["animals"]
