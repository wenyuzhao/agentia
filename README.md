# Agentia: Ergonomic LLM Agents

Ergonomic LLM Agents, with chat messages fully compatible with [Vercel AI SDK](https://ai-sdk.dev/).

# Getting Started

Run agents with tools and MCP.

```python
from agentia import Agent, MCP
from typing import Annotated

# Define a tool as a python function
def get_weather(location: Annotated[str, "The city name"]):
    """Get the current weather in a given location"""
    return { "temperature": 72 }

# Declare a MCP server:
calc = MCP(name="calculator", command="uvx", args=["mcp-server-calculator"])

# Create an agent
agent = Agent(model="openai/gpt-5-mini", tools=[get_weather, calc])

# Run the agent with the mcp
response = await agent.run("Calculate 234 ** 3")

print(response.text)

# Output: The result of 234 raised to the power of 3 is 12,812,904.
```

# The Magic Decorator

Create agent-powered magic functions.

Support both plain types and pydantic models as input and output.

```python
from agentia import magic
from pydantic import BaseModel

class Forcast(BaseModel):
    location: str
    temperature_celsius: int

@magic
async def get_weather(weather_forcast: str) -> Forcast:
    """Create weather forcase object based on the input string"""
    ...

forcast = await get_weather("The current temperature in Boston is 72°F")

print(forcast.location) # Output: Boston
print(forcast.temperature_celsius) # Output: 22
```

# Supported Parameter and Result Types

* Any types that can be passed to `pydantic.TypeAdaptor`:
    * Builtin types: `int`, `float`, `str`, `bool`, `tuple[_]`, `list[_]`, `dict[_, _]`
    * Enums: `Literal['A', 'B', ...]`, `StrEnum`, `IntEnum`, and `Enum`
    * dataclasses
* `pydantic.BaseModel` subclasses

