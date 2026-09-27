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

Tests require API keys: `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` (set via `.env` or environment).

## Architecture

### Core Flow

`Agent` (`agent.py`) is the central class. It orchestrates: model selection, tool management, conversation history, and the ReAct loop (`llm/react.py`: generate → tool calls → execute in parallel → continue). Execution modes: `agent.run()` (returns completed response), streaming via `agent.run(stream=True)`, and event-based via `agent.run(stream=True, events=True)`. MCP-based tools require `async with agent:` context manager.

### LLM Providers (`src/agentia/llm/`)

Provider abstraction with a common base (`_openai_api.py`) that speaks the **stateless OpenAI Responses API** (`/v1/responses`, always `store: false`) — the Chat Completions API is not supported. Providers: OpenAI, OpenRouter (default), Vercel, Cloudflare, Ollama. Note that Ollama and the Cloudflare `/compat` gateway do not implement `/v1/responses`. Model selector format: `[provider:][owner/]model[:mode]` (e.g., `"openai/gpt-5-nano"`, `"claude-3.5-sonnet:think"`). Default provider overridable via `AGENTIA_DEFAULT_PROVIDER` env var. Append `:think` to enable extended reasoning.

### Tool System (`src/agentia/tools/`)

`ToolSet` manages multiple tool sources:
- **Plain functions**: auto-converted using signatures and docstrings. Parameter descriptions via `Annotated[type, "description"]`.
- **Plugins** (`src/agentia/plugins/`): classes extending `Plugin` base with `@tool`-decorated methods, lifecycle hooks (`init()`), and instruction injection (`get_instructions()`).
- **MCP servers**: local (stdio) or remote (HTTP/SSE) via `MCP` class. `MCPContext` manages server lifecycle.
- **Provider-defined tools**: provider-specific built-in tools.

### Items/Models System (`src/agentia/models/`)

Pydantic models mirroring the Responses API, split across `base.py` (usage, tools, files,
annotations), `content.py` (content parts), `items.py` (the `Item` union), `options.py`
(response format) and `stream.py` (stream events).

Conversation history is a **list of `Item`**, not messages. The `Item` union mirrors the
API's `ItemField`: `Message` (`message`, with a user/assistant/system/developer role),
`Reasoning` (`reasoning` — its own item, not part of the message), `FunctionCall`
(`function_call`), `FunctionCallOutput` (`function_call_output`) and `Compaction`
(`compaction`, agentia's client-side compaction). One assistant turn yields several items.

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
`item_id`/`output_index`/`content_index`/`summary_index`. Three `agentia.*` extensions
cover what the API has no events for: `agentia.run.started` / `agentia.run.completed`
(a `run()` spans several responses) and `agentia.input_item.added` (items agentia
appends itself — client-executed tool results and `Agent.defer()` items; a
`function_call_output` is an input, so it is never a `response.output_item.*`).

`Run` (non-streaming, iterates items), `RunStream` (text/reasoning streams plus tool
items) and `RunEvents` (raw stream events) are the three `Agent.run()` return types.

### Skills (`src/agentia/plugins/skills.py`)

Discovers SKILL.md files from `.skills/`, `.agentia/skills/`, and `~/.config/agentia/skills/`. Each skill has frontmatter metadata and markdown instructions, optionally with `resources/` and `scripts/` directories.

### Magic Decorator (`src/agentia/utils/decorators.py`)

`@magic` transforms async functions into AI-powered tools with typed pydantic input/output validation.

## Key Design Patterns

- Async-first: core APIs are async, with sync wrappers where needed
- Pydantic throughout for validation and serialization
- Python 3.12+ required (uses modern type syntax)
- Build backend: hatchling; package manager: uv

## PRECAUTIONS

* Add tests when adding new features.
* Update docs when adding new features or changing the API design.
* Update CLAUDE.md when necessary
* DON'T commit, push, revert, merge, rebase unless I asked you to do so.
