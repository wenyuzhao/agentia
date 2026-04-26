import dotenv
import pytest

from agentia import Agent
from agentia.utils.commands import Commands

dotenv.load_dotenv()


def test_parse_simple():
    assert Commands.parse("/foo") == ("foo", [])
    assert Commands.parse("/foo a b c") == ("foo", ["a", "b", "c"])


def test_parse_quoted_args():
    assert Commands.parse('/greet "Hello world" Alice') == (
        "greet",
        ["Hello world", "Alice"],
    )


def test_parse_non_command():
    assert Commands.parse("foo /bar") is None
    assert Commands.parse("hello") is None
    assert Commands.parse("") is None
    assert Commands.parse("/") is None
    # Leading whitespace disqualifies the input from being a slash command.
    assert Commands.parse("   /foo bar") is None


def test_parse_unbalanced_quotes():
    assert Commands.parse('/foo "unterminated') is None


@pytest.mark.asyncio
async def test_handle_custom_handler_sync():
    agent = Agent(model="openai/gpt-5-mini")
    captured: dict = {}

    def handler(args):
        captured["args"] = args
        return "ok:" + ",".join(args)

    agent.commands.register("greet", handler)

    assert agent.commands.has("greet")
    assert agent.commands.has("/greet")
    result = await agent.commands.handle("/greet alice bob")
    assert result == "ok:alice,bob"
    assert captured["args"] == ["alice", "bob"]


@pytest.mark.asyncio
async def test_handle_custom_handler_async():
    agent = Agent(model="openai/gpt-5-mini")

    async def handler(args):
        return f"async:{len(args)}"

    agent.commands.register("/count", handler)
    result = await agent.commands.handle("/count a b c")
    assert result == "async:3"


@pytest.mark.asyncio
async def test_handle_unknown_command_passes_through():
    """Unknown slash commands are returned unchanged so the LLM can see them."""
    agent = Agent(model="openai/gpt-5-mini")
    assert await agent.commands.handle("/nope") == "/nope"


@pytest.mark.asyncio
async def test_handle_non_command_passes_through():
    """Non-command strings are returned unchanged."""
    agent = Agent(model="openai/gpt-5-mini")
    agent.commands.register("foo", lambda args: "x")
    assert await agent.commands.handle("foo bar") == "foo bar"


@pytest.mark.asyncio
async def test_handle_skill_fallback():
    agent = Agent(model="openai/gpt-5-mini", skills=["./examples"])
    result = await agent.commands.handle("/weather-skill Sydney")
    assert result is not None
    assert "Weather Skill" in result


@pytest.mark.asyncio
async def test_custom_handler_overrides_skill():
    agent = Agent(model="openai/gpt-5-mini", skills=["./examples"])
    agent.commands.register("weather-skill", lambda args: "custom")
    assert await agent.commands.handle("/weather-skill Sydney") == "custom"


@pytest.mark.asyncio
async def test_unregister():
    agent = Agent(model="openai/gpt-5-mini")
    agent.commands.register("foo", lambda args: "x")
    assert agent.commands.has("foo")
    agent.commands.unregister("foo")
    assert not agent.commands.has("foo")
    # After unregister, the command is unknown, so it passes through unchanged.
    assert await agent.commands.handle("/foo") == "/foo"


def _user_text(msg) -> str:
    """Extract concatenated text from a UserMessage, regardless of content shape."""
    content = msg.content
    if isinstance(content, str):
        return content
    return "".join(p.text for p in content if hasattr(p, "text"))


@pytest.mark.asyncio
async def test_run_skips_llm_when_handler_returns_none():
    """Custom handler that returns None: history untouched, LLM not invoked."""
    agent = Agent(model="openai/gpt-5-mini")
    called = {"n": 0}

    def handler(args):
        called["n"] += 1
        return None

    agent.commands.register("clear", handler)
    msg = await agent.run("/clear")
    assert called["n"] == 1
    # No assistant message produced and no user message added.
    assert msg.text == ""
    assert agent.history.get() == []


@pytest.mark.asyncio
async def test_run_uses_handler_output_as_prompt():
    """Custom handler returning a string: that string becomes the user prompt."""
    agent = Agent(model="anthropic/claude-haiku-4.5")
    agent.commands.register(
        "say-hi", lambda args: 'Reply with exactly the word "hello" and nothing else.'
    )
    msg = await agent.run("/say-hi")
    assert "hello" in msg.text.lower()
    history = agent.history.get()
    assert len(history) >= 2
    assert history[0].role == "user"
    assert "hello" in _user_text(history[0])


@pytest.mark.asyncio
async def test_run_streamed_skips_llm_when_handler_returns_none():
    agent = Agent(model="openai/gpt-5-mini")
    agent.commands.register("noop", lambda args: None)
    stream = agent.run("/noop", stream=True)
    items = [item async for item in stream]
    assert items == []
    assert agent.history.get() == []


@pytest.mark.asyncio
async def test_generate_object_fails_when_handler_returns_none():
    """When the handler drops the prompt, generate_object has nothing to parse."""
    from pydantic import BaseModel, ValidationError

    agent = Agent(model="openai/gpt-5-mini")
    agent.commands.register("noop", lambda args: None)

    class Out(BaseModel):
        value: int

    with pytest.raises(ValidationError):
        await agent.generate_object("/noop", Out)
