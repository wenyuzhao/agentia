# Agentia: Ergonomic LLM Agents

Ergonomic LLM Agents, with chat messages fully compatiable with [Vercel AI SDK](https://ai-sdk.dev/).

# Getting Started

Run agents with tools and MCP.

```python
from agentia import Agent, MCPServer, MCPContext
from typing import Annotated

# Define a tool as a python function
def get_weather(location: Annotated[str, "The city name"]):
    """Get the current weather in a given location"""
    return { "temperature": 72 }

# Declare a MCP server:
calc = MCPServer(name="calculator", command="uvx", args=["mcp-server-calculator"])

# Create an agent
agent = Agent(model="openai/gpt-5-mini", tools=[get_weather, calc])

# Run the agent with the tool
async with MCPContext(): # This line can be omitted if not using MCP
    response = await agent.run("What is the weather like in boston?")

print(response.text)

# Output: The current temperature in Boston is 72°F.
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

# Run agent as a REPL app

1. Create a config file at `./robo.toml`

```toml
[agent]
name = "Robo" # This is the only required field
icon = "🤖"
instructions = "You are a helpful assistant"
model = "openai/o3-mini"
plugins = ["clock"]

[mcp]
calc={ command = "uvx", args = ["mcp-server-calculator"] }
```

2. Load the agent

```python
agent = Agent.from_config("./robo.toml")
```
