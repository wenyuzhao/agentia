# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentia is an ergonomic Python library for building LLM agents with tool calling, MCP (Model Context Protocol) server integration, plugin system, and skills support. It provides a unified interface across multiple LLM providers.

## Commands

```bash
# Install dependencies
uv sync --all-extras --all-groups --all-packages --frozen

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_tools.py

# Run a single test
uv run pytest tests/test_tools.py::test_function_name

# Type checking
uv run pyright

# Lint
uv run ruff check

# Lint with auto-fix
uv run ruff check --fix

# Format
uv run ruff format
```

Tests run offline by default (mocked HTTP: `tests/mock_llm.py` is a scripted Responses API). Tests
marked `live` call real APIs and only run with `AGENTIA_LIVE_TESTS=1` plus `OPENROUTER_API_KEY`,
`OPENAI_API_KEY`, `GEMINI_API_KEY` (via `.env` or the environment).

## Architecture

### Core Flow

`Agent` (`agent.py`) is the central class: a provider, a `ToolSet`, instructions, a `History`, and the
hooks hosts override. The loop (`llm/loop.py`) works on the history, which is the only state of a
run: pending tool calls (calls without an output) run first, in parallel except calls sharing an
`exclusive()` key; otherwise `poll_inputs()` → `before_model_call()` (may raise `StopRun`) →
auto-compaction → one streamed response → items appended → `on_response()`. Every item is appended
the moment it exists, so `agent.run(None)` resumes after a crash, a cancellation or a `Suspend`.
Execution modes: `agent.run()`, `agent.run(stream=True)`, `agent.run(stream=True, events=True)`.
`async with agent:` keeps MCP connections open across runs.

### LLM Providers (`src/agentia/llm/`)

Provider abstraction with a common base (`_openai_api.py`) that speaks the **stateless OpenAI Responses API** (`/v1/responses`, always `store: false`, always streamed) over plain httpx — the Chat Completions API is not supported, and there is no OpenAI SDK dependency. A `ModelRequest` carries items, `Instruction` blocks, the compaction summary, tool schemas and options. The transport parses SSE robustly (`StreamAccumulator`), retries and falls back (`LLMOptions.max_retries` / `fallback_models`, never after output streamed), raises `LLMError`, and sends cache breakpoints (`cache_control`) after the static instructions and after the summary when the provider wants them. Providers: OpenAI, OpenRouter (default), Vercel, Cloudflare, Ollama. Note that Ollama and the Cloudflare `/compat` gateway do not implement `/v1/responses`. Model selector format: `[provider:][owner/]model[:mode]` (e.g., `"openai/gpt-5-nano"`, `"claude-3.5-sonnet:think"`). Default provider overridable via `AGENTIA_DEFAULT_PROVIDER` env var. Append `:think` to enable extended reasoning.

### Tool System (`src/agentia/tools/`)

`ToolSet` manages multiple tool sources:
- **Plain functions**: auto-converted using signatures and docstrings. Parameter descriptions via `Annotated[type, "description"]`.
- **Plugins** (`src/agentia/plugins/`): classes extending `Plugin` base with `@tool`-decorated methods, lifecycle hooks (`init()`), and instruction injection (`get_instructions()`).
- **`BaseTool` / `ModelTool`**: explicit-schema tools (pydantic args); a `ToolCall` parameter gives the call context. Tools return a value, a `ToolResult` (output, files, `is_error`, `meta`) or `Suspend`.
- **`ToolProvider`s**: tools known only at run time, reloaded each run (e.g. `MCPTools`).
- **MCP servers** (`src/agentia/mcp/`): native streamable-HTTP (`MCPClient`) and stdio (`StdioClient`) clients, OAuth helpers (`mcp/oauth.py`), the `MCPCatalog` protocol, `MCPServers` (built-in catalog over `MCP` objects) and `MCPTools` (one `mcp__<server>__<tool>` per tool, or `mcp_search` + `mcp_call` above 40).
- **Provider-defined tools**: provider-specific built-in tools.

### Items/Models System (`src/agentia/models/`)

Pydantic models mirroring the Responses API, split across `base.py` (usage, tools, files,
annotations), `content.py` (content parts), `items.py` (the `Item` union), `options.py`
(response format) and `stream.py` (stream events).

Conversation history is a **list of `Item`**, not messages. The `Item` union mirrors the
API's `ItemField`: `Message` (`message`, with a user/assistant/system/developer role),
`Reasoning` (`reasoning` — its own item, not part of the message), `FunctionCall`
(`function_call`) and `FunctionCallOutput` (`function_call_output`). One assistant turn yields several
items. Every item has a host-defined `meta` that is never sent to the provider.

### History & Compaction (`history.py`, `compaction.py`)

`History` is append-only: `items`, `compactions`, `context()` (items after the latest compaction),
`pending_tool_calls()`. Subclass and override `items` / `compactions` / `_append()` /
`_append_compaction()` to persist. A `Compaction` is a summary of the first `covers` items; its
summary goes into the system prompt between static and dynamic `Instruction`s. `split_point()`
keeps the last exchanges and never separates a call from its output. `normalize_tool_pairs()`
moves outputs next to their calls before a request.

Build messages with `Message.user()` / `.assistant()` / `.system()` / `.developer()`;
strings become `input_text`/`output_text` and a `File` becomes `input_image` /
`input_audio` / `input_file` by media type. Serialize with `item.to_responses_item()` or
`to_responses_input(items)`. `Reasoning` keeps the provider's `id`, `encrypted_content`,
`signature` and `format` so it can be replayed under `store: false`; an item without an
`id` is dropped. Structured output via `ResponseFormatJson` → Responses `text.format`.

### Streaming (`src/agentia/models/stream.py`, `src/agentia/llm/stream.py`)

Stream event `type` strings are exactly the Responses API's SSE types
(`response.output_text.delta`, `response.output_item.added`/`.done`,
`response.reasoning_summary_text.delta`, …), each carrying `sequence_number` plus
`item_id`/`output_index`/`content_index`/`summary_index`. The terminal `response.completed` carries
the complete output items. `agentia.*` extensions cover what the API has no events for:
`agentia.run.started` / `agentia.run.completed` (a `run()` spans several responses),
`agentia.input_item.added` (items agentia appends itself — tool results, `poll_inputs()` items,
`StopRun` messages) and `agentia.compaction.started` / `.completed`.

`Run` (non-streaming, iterates items), `RunStream` (text/reasoning streams plus tool
items) and `RunEvents` (raw stream events) are the three `Agent.run()` return types.

### Skills (`src/agentia/plugins/skills.py`)

`Skills(source=..., tool_name=...)` lists skills in the system prompt and provides the skill tool. A `SkillSource` (`list_skills()`, `read_skill()`) says where skills live; `LocalSkillSource` discovers SKILL.md files from `.skills/`, `.agentia/skills/`, and `~/.config/agentia/skills/`. Each skill has frontmatter metadata and markdown instructions, optionally with `resources/` and `scripts/` directories.

### Magic Decorator (`src/agentia/utils/decorators.py`)

`@magic` transforms async functions into AI-powered tools with typed pydantic input/output validation.

## Key Design Patterns

- Async-first: core APIs are async, with sync wrappers where needed
- Pydantic throughout for validation and serialization
- Python 3.12+ required (uses modern type syntax). Beware PEP 695 generics in class bodies: a field or method named `type`/`list` shadows the builtin in annotations there (use `builtins.type`)
- Core dependencies are only httpx, pydantic, python-dotenv, python-frontmatter; the Web plugin, PIL and nest-asyncio are extras
- Build backend: hatchling; package manager: uv

## PRECAUTIONS

* Add tests when adding new features.
* Update docs when adding new features or changing the API design.
* Update CLAUDE.md when necessary
* DON'T commit, push, revert, merge, rebase unless I asked you to do so.
