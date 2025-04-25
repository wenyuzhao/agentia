# Agentia: Ergonomic LLM Agent Augmented with Tools


# Getting Started

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

# Agent Config File

Agentia supports creating agents from config files:

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

# Multi-Agent Orchestration

Multi-agent orchestration is achieved by making leader/parent agents dispatching sub-tasks to their sub-agents.

```python
from agentia import Agent, CommunicationEvent, AssistantMessage

coder = Agent(
    model="openai/gpt-4o-mini",
    name="Coder",
    description="programmar",
)
reviewer = Agent(
    model="openai/gpt-4o-mini",
    name="Code Reviewer",
    description="code reviewer",
)
leader = Agent(
    model="openai/gpt-4o-mini",
    name="Leader",
    description="Leader agent",
    subagents=[
        coder,
        reviewer,
    ],
)

run = leader.run(
    "Ask your subagents to write a quicksort algorithm in python, review and improve it until it is perfect."
    events=True,
)

async for e in run:
    if isinstance(e, CommunicationEvent):
        subagent_name = coder.name if e.child == coder.id else reviewer.name
        if e.response is None:
            print(f"[DISPATCH -> {subagent_name}]")
            print(e.message)
            print("[DISPATCH END]")
            print()
        else:
            print(f"[RESPONSE <- {subagent_name}]")
            print(e.response)
            print("[RESPONSE END]")
            print()

    if isinstance(e, AssistantMessage):
        print(e.content)
        print()
```

Multi-agent orchestration in agent config files:

```toml
[agent]
name = "Leader"
# ... other fields
# Create two subagent config files in the same directory: coder.toml and reviewer.toml
subagents = ["coder", "reviewer"]
```