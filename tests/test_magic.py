from typing import Literal
from pydantic import BaseModel
from agentia import magic
from agentia.plugins import CalculatorPlugin
from enum import StrEnum


@magic
def get_content_type(extension: str) -> str:
    """
    Get the http content type for a given file extension.
    """
    ...


class DayParts(StrEnum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"


@magic
def get_day_parts(time: str) -> DayParts:
    """
    Get day part based on the time of day.
    """
    ...


class Weather(BaseModel):
    location: str
    temperature: float
    unit: Literal["celsius", "fahrenheit"]


@magic(tools=[CalculatorPlugin()])
def convert_to_fahrenheit(weather: Weather) -> Weather:
    """
    Update the weather object: convert the temperature to Fahrenheit.
    """
    ...


def test_magic1():
    assert get_content_type("txt") == "text/plain"


def test_magic2():
    assert get_day_parts("10:00 AM") == DayParts.MORNING
    assert get_day_parts("2:00 PM") == DayParts.AFTERNOON
    assert get_day_parts("10:00 PM") == DayParts.EVENING


def test_magic3():
    weather = Weather(location="San Francisco, CA", temperature=20.0, unit="celsius")
    assert convert_to_fahrenheit(weather) == Weather(
        location="San Francisco, CA", temperature=68.0, unit="fahrenheit"
    )
