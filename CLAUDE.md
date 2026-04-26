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

`Agent` (`agent.py`) is the central class. It orchestrates: model selection, tool management, conversation history, and the agentic loop (`llm/agentic.py`: generate → tool calls → execute in parallel → continue). Execution modes: `agent.run()` (returns completed response), streaming via `agent.run(stream=True)`, and event-based via `agent.run(stream=True, events=True)`. MCP-based tools require `async with agent:` context manager.

### LLM Providers (`agentia/llm/`)

Provider abstraction with a common OpenAI-compatible base (`_openai_api.py`). Providers: OpenAI, OpenRouter (default), Vercel, Cloudflare, Ollama, Gemini Live. Model selector format: `[provider:][owner/]model[:mode]` (e.g., `"openai/gpt-5-nano"`, `"claude-3.5-sonnet:think"`). Default provider overridable via `AGENTIA_DEFAULT_PROVIDER` env var. Append `:think` to enable extended reasoning.

### Tool System (`agentia/tools/`)

`ToolSet` manages multiple tool sources:
- **Plain functions**: auto-converted using signatures and docstrings. Parameter descriptions via `Annotated[type, "description"]`.
- **Plugins** (`agentia/plugins/`): classes extending `Plugin` base with `@tool`-decorated methods, lifecycle hooks (`init()`), and instruction injection (`get_instructions()`).
- **MCP servers**: local (stdio) or remote (HTTP/SSE) via `MCP` class. `MCPContext` manages server lifecycle.
- **Provider-defined tools**: provider-specific built-in tools.

### Message/Models System (`agentia/models/`)

Pydantic-based message models. Types: SystemMessage, UserMessage, AssistantMessage, ToolMessage. Multi-modal content support (text, images, audio, video, files). Structured output via pydantic BaseModel in `ResponseFormatJson`.

### Streaming (`agentia/llm/stream.py`)

`ChatCompletionStream` for async message iteration; `ChatCompletionEvents` for granular stream part events (text-start/delta/end, reasoning-start/delta/end, audio events).

### Skills (`agentia/plugins/skills.py`)

Discovers SKILL.md files from `.skills/`, `.agentia/skills/`, and `~/.config/agentia/skills/`. Each skill has frontmatter metadata and markdown instructions, optionally with `resources/` and `scripts/` directories.

### Magic Decorator (`agentia/utils/decorators.py`)

`@magic` transforms async functions into AI-powered tools with typed pydantic input/output validation.

### Live Sessions (`agentia/live.py`)

WebSocket-based real-time bidirectional streaming for audio/video/text (currently via Gemini Live API).

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
