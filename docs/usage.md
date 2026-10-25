
# Agentia Library Usage Guide

```bash
pip install agentia            # core: httpx + pydantic
pip install "agentia[all]"     # + the Web plugin (tavily, markdownify), PIL images for @magic, notebooks
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

Supported API providers: `openai`, `openrouter`, `vercel`, `cloudflare`, `ollama` (all over the
Responses API). Override the default with the `AGENTIA_DEFAULT_PROVIDER` env var.

For full control (endpoint, key, headers, a shared HTTP client), build the provider yourself:

```python
import httpx
from agentia import Agent
from agentia.llm.providers.openrouter import OpenRouter

http = httpx.AsyncClient()
agent = Agent(
    provider=OpenRouter(
        "anthropic/claude-sonnet-4.6",
        api_key="sk-or-…",
        base_url="https://openrouter.ai/api/v1",  # default; OPENROUTER_BASE_URL also works
        client=http,
        headers={"x-title": "my-app"},
    ),
    client=http,
)
```

The transport is plain httpx: it always streams, never stores (`store: false`), retries 408/429/5xx
and network errors with jittered backoff (`LLMOptions.max_retries`, default 3), then tries
`LLMOptions.fallback_models` (never for 401/402/403), and never retries once output has streamed.
Failures raise `agentia.LLMError`. OpenRouter's `usage.cost` lands in `Usage.cost`.

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
- `tools` (list) - Functions, Plugins, `BaseTool`s, `ToolProvider`s, MCP servers
- `instructions` (str | Instruction | Callable | list) - System prompt blocks
- `options` (LLMOptions) - LLM configuration
- `skills` (bool | list | Skills) - Skill loading config
- `provider` (Provider) - A configured provider (instead of `model`)
- `history` (History) - The conversation store (subclass it to persist)
- `compaction` (bool | CompactionOptions) - Automatic compaction
- `client` (httpx.AsyncClient) - A shared HTTP client for model calls

### Running the Agent

**Await for full response:**

```python
response = await agent.run("What is the weather?")
print(response.text)
```

**Async iteration over items:**

A turn is a sequence of items, not a single message, so check `item.type`:

```python
async for item in agent.run("What is the weather?"):
    match item.type:
        case "message" if item.role == "assistant":
            print(item.text)
        case "reasoning":
            print("[REASONING]", item.display_text)
        case "function_call":
            print("[CALL]", item.name, item.input)
        case "function_call_output":
            print("[RESULT]", item.text)
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

Event `type`s are the Responses API's own SSE type strings:

```python
run = agent.run("Hello", stream=True, events=True)
async for event in run:
    if event.type == "response.output_text.delta":
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
from agentia import Instruction

# At construction
agent = Agent(instructions="Be concise.")
agent = Agent(
    instructions=[
        "Be concise.",
        lambda: Instruction(text=f"Now: {datetime.now()}", dynamic=True),
    ]
)

# After construction (sync or async callables work)
agent.add_instructions("Additional context.")
```

The system prompt is assembled as: static blocks (yours, then plugins' and tool providers'),
then the compaction summary, then `dynamic=True` blocks. Keeping changing content in dynamic
blocks keeps the prefix cacheable; models that need explicit cache breakpoints (OpenRouter's
Anthropic and Gemini models, or `LLMOptions(cache_control=True)`) get one after the static blocks
and one after the summary.

### Conversation History

History is an append-only list of Responses API items (see [Items](#11-items--the-responses-api))
plus the compactions over it:

```python
agent.history.add(Message.user("Hi"))  # Add one or more items
agent.history.items  # every item ever added
agent.history.context()  # what the model sees after the latest summary
agent.history.compaction  # the latest Compaction, or None
agent.history.pending_tool_calls()  # calls without an output (a suspended or interrupted run)
agent.history.load(items, compactions)  # Replace everything
agent.history.clear()
```

To persist a conversation, subclass `History` and override the storage: the `items` and
`compactions` properties, `_append(items)` and `_append_compaction(c)`. Every response item and
tool result is appended the moment it exists, so a persisted history is always resumable
(see [Hosting](#6-hosting-an-agent)).

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
        max_steps=50,  # stop a run after 50 model calls (finish reason "max-steps")
        max_retries=3,
        fallback_models=["openai/gpt-5.4-mini"],
        timeout=600,
        cache_control=None,  # None: the provider decides
    ),
)

# Override per-call
response = await agent.run("Hello", options=LLMOptions(temperature=0))
```

`presence_penalty`, `frequency_penalty`, `seed` and `stop_sequences` have no
first-class field in the OpenAI Responses API. They are only forwarded (in the
request's extra body) when you set them explicitly, so they reach gateways that
still accept them and are otherwise absent from the request.

`ReasoningOptions.effort` and `ReasoningOptions.max_tokens` are mutually
exclusive; `effort` wins when both are set.

## 1.1 Items & the Responses API

Agentia speaks the OpenAI **Responses API** exclusively, in its stateless form —
nothing is stored server-side (`store: false`) and the full conversation is sent on
every request. Accordingly, the conversation history is a **list of items**
(`agent.history.items`), mirroring the API's `ItemField` union rather than a list of
chat messages:

| Item | `type` | What it is |
| --- | --- | --- |
| `Message` | `message` | Text content with a `role` of user / assistant / system / developer |
| `Reasoning` | `reasoning` | The model's reasoning, its own item — not part of the message |
| `FunctionCall` | `function_call` | A tool call the model requested |
| `FunctionCallOutput` | `function_call_output` | A tool result, an *input* for the next request |

Every item has a `meta` field for host-defined metadata (sequence numbers, timestamps, …);
agentia never interprets it and never sends it to the provider.

One assistant turn therefore produces **several** items, in order: a `reasoning` item,
then a `message` and/or one `function_call` per tool the model wants to run.

### Building Messages

`Message` has a helper per role. Strings become `input_text` (or `output_text` for
assistant messages), and a `File` becomes `input_image`, `input_audio` or `input_file`
according to its media type:

```python
from agentia import Message, File

Message.user("What is in this image?")
Message.user(
    [
        "What is this animal?",
        File(data="https://example.com/cat.png", media_type="image/png"),
    ]
)
Message.assistant("Hello!")
Message.system("Be concise.")
Message.developer("Prefer SI units.")
```

Content parts can also be constructed directly — `InputText`, `InputImage`,
`InputFile`, `InputAudio`, `OutputText`, `Refusal`:

```python
from agentia import Message, InputText, InputImage

Message(
    role="user",
    content=[InputText(text="Compare these"), InputImage(image_url="https://…")],
)
```

Useful accessors: `Message.text` (all text concatenated), `Message.refusal`,
`Message.parse(SomeModel)` for structured output, `Reasoning.text` /
`.summary_text` / `.display_text`, `FunctionCall.input` (parsed `arguments`), and
`FunctionCallOutput.text`.

### Reasoning Round-Trips

Because nothing is stored server-side, a reasoning item can only be replayed on the
next request if it keeps the provider's own identity:

```python
item.id  # Responses output item id, e.g. "rs_..."
item.content  # [ReasoningText(...)] — raw reasoning
item.summary  # [SummaryText(...)]  — summary blocks
item.encrypted_content  # opaque payload returned because store=false
item.signature  # provider signature over the reasoning content
item.format  # e.g. "openai-responses-v1"
```

A `Reasoning` item with no `id` is dropped when serializing, since the provider would
reject it. Use `to_responses_input(items)` to serialize a list yourself.

### Compaction

`Compaction` plays the role of the API's `compaction` item, but agentia compacts
client-side and never rewrites the history: a `Compaction` is a plain-text `summary` of the
first `covers` items. The model sees the latest summary (in the system prompt) followed by
`history.context()`.

```python
from agentia import CompactionOptions

agent = Agent(
    compaction=CompactionOptions(
        threshold=0.6,  # compact before a model call above 60% of the context window…
        max_threshold_tokens=100_000,  # …capped at 100k tokens
        keep_exchanges=4,  # keep the last 4 user↔assistant exchanges verbatim
        keep_steps=6,  # inside one huge exchange, keep the last 6 model responses
        model="openai/gpt-5.4-nano",  # optional cheaper summarizer
    )
)

c = await agent.compact()  # now (None if nothing is old enough)
c = await agent.compact(
    "the database migration", keep_exchanges=0
)  # fold everything, with a focus
```

The split only happens before a user message or a model response, so a function call is
never separated from its output. Override `Agent.render_compaction_input`,
`Agent.render_summary`, `Agent.is_turn_start` or `Agent.on_compaction` to customize it.

### Stream Events

Event `type` strings are exactly the Responses API's, and every event carries a
`sequence_number`, so you can match on them directly:

`response.created` · `response.in_progress` · `response.completed` ·
`response.incomplete` · `response.failed` · `response.output_item.added` / `.done` ·
`response.content_part.added` / `.done` · `response.output_text.delta` / `.done` ·
`response.output_text.annotation.added` · `response.refusal.delta` / `.done` ·
`response.reasoning_text.delta` / `.done` ·
`response.reasoning_summary_text.delta` / `.done` ·
`response.reasoning_summary_part.added` / `.done` ·
`response.function_call_arguments.delta` / `.done` · `error`

Two things the Responses API has no events for are namespaced under `agentia.*`:

| Event | Meaning |
| --- | --- |
| `agentia.run.started` | A whole `run()` began. One per `run()`. |
| `agentia.run.completed` | A whole `run()` finished, carrying total `usage` and `finish_reason`. |
| `agentia.input_item.added` | Agentia appended an input item — a client-executed tool result, an item from `Agent.poll_inputs()` / `Agent.defer()`, or a `StopRun` message. |
| `agentia.compaction.started` / `.completed` | An automatic compaction ran inside the run. |

`response.output_item.*` covers only what the model produced, which is why a
client-executed tool result is *not* one: a `function_call_output` is an input to the
next request, so it arrives as `agentia.input_item.added`.

A single `run()` spans one `response.created` … `response.completed` cycle per
iteration of the ReAct loop:

```
agentia.run.started
  response.created
    response.output_item.added   (reasoning)
    response.reasoning_summary_text.delta …
    response.output_item.done    (reasoning)
    response.output_item.added   (function_call)
    response.function_call_arguments.delta …
    response.output_item.done    (function_call)
  response.completed
  agentia.input_item.added       (function_call_output — agentia ran the tool)
  response.created               (next iteration)
    …
  response.completed
agentia.run.completed
```

## 2. Function Tools & Plugins

### Plain Functions as Tools

Any typed function works as a tool. Use `Annotated` for parameter descriptions. The docstring becomes the tool description.

```python
from typing import Annotated, Literal


def get_weather(
    location: Annotated[str, "The city and state, e.g. San Francisco, CA"],
    unit: Annotated[
        Literal["celsius", "fahrenheit"], "Temperature unit"
    ] = "fahrenheit",
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

### Tools with Explicit Schemas

`ModelTool` takes a pydantic arguments model (validation errors go back to the model);
subclass `BaseTool` for a raw JSON schema. A parameter annotated `ToolCall` (or `Agent`) is
injected rather than shown to the model:

```python
from pydantic import BaseModel
from agentia import ModelTool, ToolCall


class WriteArgs(BaseModel):
    path: str
    content: str


async def write(ctx: ToolCall, a: WriteArgs) -> str:
    ...  # ctx.agent, ctx.call_id
    return f"wrote {a.path}"


# Calls with the same exclusive key run one at a time; others in a batch run in parallel.
tool = ModelTool(
    "write_file", "Write a file", WriteArgs, write, exclusive=lambda a: a.path
)
```

A tool may also return `Suspend()` to pause the run (see [Hosting](#6-hosting-an-agent)).
Failures never abort a run: an exception, an unknown tool or invalid arguments become an
error output for the model.

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
        is_error=False,
        meta={"path": "/tmp/chart.png"},  # copied onto the FunctionCallOutput
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
- `get_instructions()` to inject system prompt text (may be async)
- `tools()` to add `BaseTool`s besides the `@tool` methods
- `self.agent` to access the parent agent
- Both sync and async `@tool` methods supported

### Lazy Tool Loading

```python
agent = Agent(model="openai/gpt-5.4-nano")
response = await agent.run("Hi!")  # No tools yet

agent.tools.add(get_weather)  # Add tool later
agent.tools.add(WeatherPlugin())  # Add plugin later
response = await agent.run("Weather in Boston?")
```

### Built-in Plugins

Available in `agentia.plugins`: `System` (`Bash`, `LoadFile`), `Web` (`WebSearch`, `WebFetch`;
needs `agentia[web]`) and `Skills`.

```python
from agentia.plugins import System

agent = Agent(model="openai/gpt-5.4-nano", tools=[System()])
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

Optional parameters: `env` (dict), `cwd` (str/Path), `timeout` (seconds per call, default 300).

### Remote Server (streamable HTTP)

```python
MCP(name="api", url="https://api.example.com/mcp")
MCP(
    name="api", url="https://api.example.com/mcp", headers={"x-team": "1"}, auth="token"
)
MCP(name="api", url="https://api.example.com/mcp", auth=my_httpx_auth, client=http)
```

`auth` is a bearer token or an `httpx.Auth`. The legacy SSE transport is not supported.
`agentia.mcp.oauth` has the MCP authorization steps (metadata discovery, dynamic client
registration, PKCE, code exchange, refresh) for hosts that run their own login flow.

Each server's tools appear as `mcp__<server>__<tool>`; `tool.metadata["mcp"]["destructive"]`
reflects the tool's annotations, for approval policies. With more than 40 tools the agent gets
`mcp_search` + `mcp_call` instead of the whole list.

### Lifecycle

Connections open on first use. `async with agent:` keeps them open across runs and closes them
at the end; without it, they are closed after each run.

```python
async with agent:
    response = await agent.run("Calculate 234 ** 3")
    print(response.text)
```

### Host-managed MCP

To manage connections yourself (OAuth, secrets, caching), implement the `MCPCatalog` protocol —
`async catalog() -> (servers, tools)` and `async call(server, tool, arguments)` — and expose it
with `MCPTools`:

```python
from agentia import MCPTools

tools = MCPTools(
    my_catalog,
    mode="auto",  # "direct" | "search" | "auto"
    allow=["mcp__github__*"],  # optional allowlist
    on_auth_required=lambda server, url: notify_user(server, url),
)
agent = Agent(tools=[tools])
```

`agentia.mcp.MCPClient` (streamable HTTP, with a pluggable `send` for custom transports) and
`StdioClient` are available for building one.

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

Skills that live elsewhere (a sandbox, a database) come from a custom `SkillSource`:

```python
from agentia import Skill, Skills, SkillSource


class RemoteSkills(SkillSource):
    async def list_skills(self) -> list[Skill]:
        return [
            Skill(name="pdf", description="PDF tools", path="/opt/skills/pdf/SKILL.md")
        ]

    async def read_skill(self, skill: Skill, args: list[str] | None = None) -> str:
        return await fetch(skill.path)


agent = Agent(tools=[Skills(source=RemoteSkills(), tool_name="skill")])
```

The plugin lists the skills in the system prompt and the tool returns
`# Skill <name> (<folder>)` followed by the instructions, so relative references resolve
against the skill's folder.

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
- `disable-model-invocation: true` — hide the skill from the model; it won't appear in system instructions and the skill tool refuses to execute it. Useful for skills meant only for direct user invocation.
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
result = await agent.commands.handle('/greet "Alice"')  # -> "Hello, Alice!"
result = await agent.commands.handle("/clear")  # -> "history cleared"
```

Resolution order:
1. Custom handler registered with `agent.commands.register(name, handler)`.
2. A user-invocable skill (i.e. `user-invocable: true`, the default) with the same name.
3. Otherwise `handle()` returns the input unchanged, so it reaches the model as text.

Handlers receive the positional argument list parsed via `shlex` (so quoted
arguments work). They may return `str` or `None`, sync or async. Use
`agent.commands.unregister(name)` to remove a handler, `await agent.commands.has(name)`
to check whether a name resolves, and `Commands.parse(input)` to parse a
command string without dispatching.

### Slash Commands in `run` and `generate_object`

`agent.run(prompt)` and `agent.generate_object(prompt, type)` automatically
dispatch slash commands when `prompt` is a string starting with `/`:

- If a handler/skill returns a string, that string is used as the prompt sent
  to the LLM (the original `/cmd` text is not stored in history).
- If a handler returns `None`, the LLM is **not invoked**. `run()` returns an empty
  assistant `Message`; `generate_object()` raises `ValueError`.
- If nothing matches, the text is sent to the model as is.

```python
agent.commands.register("clear", lambda args: (agent.history.clear(), None)[1])

await agent.run("/clear")  # handler runs; LLM is skipped
await agent.run("/weather Sydney")  # skill markdown becomes the prompt
```

## 6. Hosting an Agent

agentia is built to be embedded in long-running hosts (bots, servers) that must survive
restarts and wait for people. The history is the only state of a run: each response item and
each tool result is appended the moment it exists. `agent.run(None)` continues from the
history — it executes pending tool calls (calls without an output), otherwise calls the model.

### The loop

1. Pending tool calls run (in parallel, except calls sharing an `exclusive()` key). Each output
   is appended as soon as it exists.
2. Otherwise: `poll_inputs()` items are appended, `before_model_call()` may stop the run,
   the context is compacted if it is over the threshold, one response is streamed and its items
   are appended, then `on_response()` runs.
3. The run ends when the model answers without tool calls and `poll_inputs(after_answer=True)`
   returns nothing.

`run.finish_reason` is the model's (`stop`, `length`, …) or one of `suspended`, `stopped`,
`max-steps`.

### Hooks

Subclass `Agent` and override:

| Hook | Use |
| --- | --- |
| `poll_inputs(after_answer)` | Steering: messages that arrived mid-run (default: `defer()`ed items) |
| `before_model_call(step)` | Budgets and access checks; raise `StopRun("message")` to end the run |
| `prepare_input(items)` | Shape what the model sees (shorten old results, attach images); default pairs calls with outputs |
| `on_response(items, response)` | Accounting, delivery; `response.usage`, `response.model` |
| `before_tool_call(call, tool)` | Approvals and restart safety: return a result, `Suspend()`, or `None` to run |
| `after_tool_call(call, output)` | Replace the output before it is recorded (e.g. offload large output) |
| `on_compaction(compaction, replaced)` | Before a compaction is committed (write transcripts, record cost) |
| `render_summary(c)`, `render_compaction_input(...)`, `is_turn_start(item)` | Compaction rendering |
| `context_window()` | The window used for the compaction threshold |
| `new_call_id(call, seen)` | Rewrite provider call ids (default: only empty or duplicate ones) |

### Suspension

```python
from agentia import Agent, Suspend


class Approving(Agent):
    approved: set[str] = set()

    async def before_tool_call(self, call, tool):
        if call.name == "rm" and call.call_id not in self.approved:
            await ask_user(call)  # e.g. send buttons
            return Suspend("approval")
        return None


run = agent.run("clean up")
await run
if run.suspended:
    ...  # later, when the user approves:
    agent.approved.add(call_id)
    await agent.run(None)  # the call runs, then the model continues
```

A tool can suspend too, e.g. while it waits for a sub-agent; when the result is ready, append
its `FunctionCallOutput` to the history and call `agent.run(None)`. A tool that may run again on
resume must be idempotent by `call_id`. Cancelling the task that runs `agent.run()` leaves the
history resumable in the same way.
