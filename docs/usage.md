
# Agentia Library Usage Guide

```bash
pip install agentia
```

```python
from agentia import Agent, MCP, Plugin, tool, magic, LLMOptions, ToolResult
```

## 1. Agent Class

### Model Selector

Format: `api_provider:provider/model`. Default API provider is `openrouter`.

```python
# These are equivalent (openrouter is the default API provider)
agent = Agent(model="openrouter:openai/gpt-5.4-nano")
agent = Agent(model="openai/gpt-5.4-nano")

# Enable reasoning with :think suffix
agent = Agent(model="openai/gpt-5.4-nano:think")
```

Supported API providers: `openai`, `openrouter`, `vercel`, `cloudflare`, `qwen`, `chutes`, `fireworks`, `ollama`. Override default with `AGENTIA_DEFAULT_PROVIDER` env var.

### Creating an Agent

```python
agent = Agent(
    model="openai/gpt-5.4-nano",
    tools=[my_tool, MyPlugin(), mcp_server],
    instructions="You are a helpful assistant.",
    options=LLMOptions(temperature=0.7, max_output_tokens=1000),
    skills=True,
)
```

Constructor parameters:
- `model` (str) - Model selector
- `tools` (list) - Functions, Plugins, MCP servers
- `instructions` (str | Callable | list) - System prompt(s)
- `options` (LLMOptions) - LLM configuration
- `skills` (bool | list | Skills) - Skill loading config

### Running the Agent

**Await for full response:**

```python
response = await agent.run("What is the weather?")
print(response.text)
```

**Async iteration over messages:**

```python
async for msg in agent.run("What is the weather?"):
    if msg.role == "assistant":
        print(msg.text)
```

**Streaming text chunks:**

```python
from agentia import MessageStream, ReasoningStream

run = agent.run("Tell me a story", stream=True)
async for stream in run:
    if isinstance(stream, MessageStream):
        async for chunk in stream:
            print(chunk, end="", flush=True)
    elif isinstance(stream, ReasoningStream):
        reasoning = await stream
        print("[REASONING]", reasoning)
```

**Event-based streaming:**

```python
run = agent.run("Hello", stream=True, events=True)
async for event in run:
    if event.type == "text-delta":
        print(event.delta, end="")
```

### Structured Output

```python
from pydantic import BaseModel, Field
from enum import StrEnum

class HairColor(StrEnum):
    BLACK = "black"
    BROWN = "brown"
    BLONDE = "blonde"

class Person(BaseModel):
    name: str
    age: int
    hair_color: HairColor
    hobbies: list[str]
    location: str = Field(..., description="City and country")

result = await agent.generate_object("Get details about Einstein", Person)
assert isinstance(result, Person)
```

### Magic Decorator

Turn any function into an AI-powered function with typed returns:

```python
from agentia import magic, ImageUrl

@magic
async def summarise(text: str) -> str:
    """Summarise the given text to one sentence."""
    ...

print(await summarise("Long article text here..."))

@magic
async def classify(text: str) -> Literal["positive", "negative", "neutral"]:
    """Classify sentiment."""
    ...

@magic(model="openai/gpt-5.4-nano", tools=[Calculator()])
async def calculate(expression: str) -> int:
    """Calculate the given expression."""
    ...

# Image inputs
@magic
async def describe(image: ImageUrl) -> str:
    """Describe the image."""
    ...

print(await describe(ImageUrl("https://example.com/photo.jpg")))
```

Supported return types: `str`, `int`, `float`, `bool`, `list`, `dict`, `BaseModel`, `Enum`, `Literal`.

### Instructions

```python
# At construction
agent = Agent(instructions="Be concise.")
agent = Agent(instructions=["Be concise.", lambda: f"Today is {date.today()}"])

# After construction
agent.add_instructions("Additional context.")
```

### Conversation History

```python
agent.history.clear()                        # Clear messages, keep instructions
agent.history.clear(clear_instructions=True)  # Clear everything
agent.history.add(user_message)               # Add a message
agent.history.load(messages)                  # Replace all messages
messages = agent.history.get()                # Get full history
```

### LLM Options

```python
from agentia import LLMOptions, ReasoningOptions

agent = Agent(
    model="openai/gpt-5.4-nano",
    options=LLMOptions(
        temperature=0.7,
        max_output_tokens=2000,
        top_p=0.9,
        presence_penalty=0.1,
        frequency_penalty=0.1,
        seed=42,
        parallel_tool_calls=True,
        reasoning=ReasoningOptions(enabled=True, effort="high"),
    ),
)

# Override per-call
response = await agent.run("Hello", options=LLMOptions(temperature=0))
```

## 2. Function Tools & Plugins

### Plain Functions as Tools

Any typed function works as a tool. Use `Annotated` for parameter descriptions. The docstring becomes the tool description.

```python
from typing import Annotated, Literal

def get_weather(
    location: Annotated[str, "The city and state, e.g. San Francisco, CA"],
    unit: Annotated[Literal["celsius", "fahrenheit"], "Temperature unit"] = "fahrenheit",
):
    """Get the current weather in a given location"""
    return {"location": location, "temperature": "72", "unit": unit}

agent = Agent(model="openai/gpt-5.4-nano", tools=[get_weather])
response = await agent.run("What's the weather in Boston?")
```

Async functions work too:

```python
async def search_db(query: Annotated[str, "Search query"]):
    """Search the database"""
    results = await db.search(query)
    return results
```

### @tool Decorator

Use for custom name, description, or metadata:

```python
from agentia import tool

@tool(name="weather_lookup", description="Look up weather data")
def get_weather(location: Annotated[str, "City name"]):
    return {"temperature": 72}
```

### ToolResult with Files

```python
from agentia import ToolResult, File

@tool
def generate_chart(data: Annotated[str, "CSV data"]):
    """Generate a chart from data"""
    # ... generate chart_bytes ...
    return ToolResult(
        output="Chart generated",
        files=[File(data=chart_bytes, media_type="image/png")],
    )
```

### Plugins

Extend `Plugin` and decorate methods with `@tool`:

```python
from agentia import Agent, Plugin, tool

class WeatherPlugin(Plugin):
    @tool
    def get_forecast(
        self,
        location: Annotated[str, "The city and state, e.g. San Francisco, CA"],
    ):
        """Get the current weather forecast"""
        return {"forecast": ["sunny", "windy"]}

    @tool
    def get_temperature(
        self,
        location: Annotated[str, "The city and state"],
        unit: Literal["celsius", "fahrenheit"] | None = "fahrenheit",
    ):
        """Get the current temperature"""
        return {"temperature": "72", "unit": unit}

agent = Agent(model="openai/gpt-5.4-nano", tools=[WeatherPlugin()])
response = await agent.run("Weather in Boston?")
```

Plugin features:
- `NAME` class var for custom plugin name
- `async init()` for async setup (e.g., API auth)
- `get_instructions()` to inject system prompt text
- `self.agent` to access the parent agent
- Both sync and async `@tool` methods supported

### Lazy Tool Loading

```python
agent = Agent(model="openai/gpt-5.4-nano")
response = await agent.run("Hi!")  # No tools yet

agent.tools.add(get_weather)       # Add tool later
agent.tools.add(WeatherPlugin())   # Add plugin later
response = await agent.run("Weather in Boston?")
```

### Built-in Plugins

Available in `agentia.plugins`: `Calculator`, `Clock`, `CodeRunner`, `Memory`, `Search`, `Web`.

```python
from agentia.plugins import Calculator

agent = Agent(model="openai/gpt-5.4-nano", tools=[Calculator()])
```

## 3. MCP Servers

### Local Server (stdio)

```python
from agentia import Agent, MCP

agent = Agent(
    model="openai/gpt-5.4-nano",
    tools=[
        MCP(name="calculator", command="uvx", args=["mcp-server-calculator"]),
    ],
)
```

Optional parameters: `env` (dict), `cwd` (str/Path), `timeout` (int).

### Remote Server (HTTP/SSE)

```python
MCP(name="api", type="http", url="https://api.example.com/mcp")
MCP(name="api", type="sse", url="https://api.example.com/sse", headers={"Authorization": "Bearer ..."})
MCP(name="api", type="streamable-http", url="https://...", auth="oauth")
```

### Context Manager Requirement

When using MCP tools, wrap in `async with agent:`:

```python
agent = Agent(
    model="openai/gpt-5.4-nano",
    tools=[MCP(name="calc", command="uvx", args=["mcp-server-calculator"])],
)

async with agent:
    response = await agent.run("Calculate 234 ** 3")
    print(response.text)
```

## 4. Skills

### Enabling Skills

```python
# Auto-discover from default paths
agent = Agent(model="openai/gpt-5.4-nano", skills=True)

# Custom search paths
agent = Agent(model="openai/gpt-5.4-nano", skills=["./my-skills", "./examples"])

# Full control
from agentia import Skills
agent = Agent(model="openai/gpt-5.4-nano", skills=Skills(search_paths=["./skills"]))
```

Default search paths: `.skills/`, `.agentia/skills/`, `~/.config/agentia/skills/`.

### Skill Format

Skills follow the open SKILL.md standard. See https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices for the full specification.

Basic structure:

```
my-skill/
  SKILL.md          # Required: YAML frontmatter (name, description) + markdown instructions
  resources/        # Optional: reference files (.md, .json, .yaml, .csv, .txt)
  scripts/          # Optional: executable scripts (.py, .sh, .js)
```

Optional frontmatter flags:
- `disable-model-invocation: true` — hide the skill from the model; it won't appear in system instructions and the `Skill` tool refuses to execute it. Useful for skills meant only for direct user invocation.
- `user-invocable: false` — exclude the skill from the user-invocable set (e.g. slash-command surfaces) while still letting the model discover and execute it. Defaults to `true`.

Example `SKILL.md`:

```markdown
---
name: weather-skill
description: Get the weather and temperature of a given location
---

# Weather Skill

Get the weather and temperature of a given location.

## Getting Started

1. Identify the location the user is interested in.
2. Run the get-weather script with the location.

## Examples

python scripts/get-weather.py "Sydney"
```

### Substitutions

When a skill is executed, its `SKILL.md` content is processed for the following inline substitutions before being returned:

**Bash command substitution** — `` !`cmd` `` runs `cmd` through the shell (with the skill directory as cwd, 30s timeout) and is replaced with the captured stdout.

```markdown
Current date: !`date +%Y-%m-%d`
Files in skill: !`ls`
```

**Argument substitutions** — resolved against arguments passed to the skill (and the frontmatter `arguments` list, when present):

| Syntax | Meaning |
| --- | --- |
| `$ARGUMENTS` | All arguments joined by spaces |
| `$ARGUMENTS[N]` | The Nth argument (0-indexed); empty if out of range |
| `$N` | Shorthand for `$ARGUMENTS[N]` |
| `$name` | Named argument, resolved positionally against the frontmatter `arguments` list |
| `${VAR}` | Environment variable; left untouched if unset |
| `${CLAUDE_SKILL_DIR}` / `${SKILL_DIR}` | Absolute path to the skill's directory |

Example `SKILL.md` using named arguments:

```markdown
---
name: greet
description: Greet a user
arguments: [name, greeting]
---

$greeting, $name! (raw: $ARGUMENTS)
```

Calling the skill with `/greet "Alice" "Hello"` yields `Hello, Alice! (raw: Alice Hello)`.

## 5. Slash Commands

`agent.commands` dispatches `/name arg0 arg1 ...` strings to either a custom
handler or a user-invocable skill.

```python
agent = Agent(model="openai/gpt-5.4-nano", skills=True)

# Register a custom handler. Sync or async; receives the parsed args list.
agent.commands.register("greet", lambda args: f"Hello, {args[0]}!")

async def clear(args):
    agent.history.clear()
    return "history cleared"

agent.commands.register("/clear", clear)  # leading slash is optional

# Dispatch:
result = await agent.commands.handle('/greet "Alice"')   # -> "Hello, Alice!"
result = await agent.commands.handle("/clear")           # -> "history cleared"
```

Resolution order:
1. Custom handler registered with `agent.commands.register(name, handler)`.
2. A user-invocable skill (i.e. `user-invocable: true`, the default) with the same name.
3. Otherwise `handle()` returns `None` and does nothing.

Handlers receive the positional argument list parsed via `shlex` (so quoted
arguments work). They may return `str` or `None`, sync or async. Use
`agent.commands.unregister(name)` to remove a handler, `agent.commands.has(name)`
to check whether a name resolves, and `Commands.parse(input)` to parse a
command string without dispatching.

### Slash Commands in `run` and `generate_object`

`agent.run(prompt)` and `agent.generate_object(prompt, type)` automatically
dispatch slash commands when `prompt` is a string starting with `/`:

- If a handler/skill returns a string, that string is used as the prompt sent
  to the LLM (the original `/cmd` text is not stored in history).
- If a handler returns `None`, or no handler/skill matches, the LLM is **not
  invoked**. `run()` returns an empty `AssistantMessage`; `generate_object()`
  raises `ValueError`.

```python
agent.commands.register("clear", lambda args: (agent.history.clear(), None)[1])

await agent.run("/clear")          # handler runs; LLM is skipped
await agent.run("/weather Sydney") # skill markdown becomes the prompt
```
