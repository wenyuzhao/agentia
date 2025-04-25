# Agentia: Ergonomic LLM Agent Augmented with Tools


## Getting Started

```python
from agentia import Agent
from typing import Annotated

# Define a tool as a python function
def get_weather(location: Annotated[str, "The city name"]):
    """Get the current weather in a given location"""
    return { "temperature": 72 }

# Create an agent
agent = Agent(tools=[get_weather])

# Run the agent with the tool
response = await agent.run("What is the weather like in boston?")

print(response)

# Output: The current temperature in Boston is 72°F.
```

## Create an Agent from a Config File

1. Create a config file at `./alice.toml`

```toml
[agent]
name = "Alice" # This is the only required field
icon = "👩"
instructions = "You are a helpful assistant"
model = "openai/o3-mini"
plugins = ["calc", "clock", "web"]
```

2. In your python code:

```python
agent = Agent.load_from_config("./alice.toml")
```

3. Alternatively, start a REPL:

```bash
uvx agentia repl alice
```
