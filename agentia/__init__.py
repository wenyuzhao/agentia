from __future__ import annotations

import logging
from pathlib import Path

import dotenv

from .agent import Agent
from .models import *
from .utils.decorators import magic, ImageUrl
from .utils.decorators import tool
from .plugins import Plugin, PluginInitError
from .plugins.skills import Skills, Skill
from . import plugins
from .tools import Tool, Tools, MCP, MCPContext, ToolResult
from .history import History
from .llm import LLMOptions, ReasoningOptions
from .llm.completion import ChatCompletion
from .llm.stream import (
    TextStream,
    ReasoningStream,
    MessageStream,
    ChatCompletionStream,
    ChatCompletionEvents,
)

LOGGER = logging.getLogger("agentia")


def init_logging(level: logging._Level = logging.INFO):
    logging.basicConfig(
        level=level, format="[%(asctime)s][%(levelname)-8s][%(name)s] %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def load_dotenv(
    *, dotenv_path: str | Path | None = None, cwd: bool = True, override: bool = True
):
    """Load environment variables from a .env file. By default, it looks for a .env file in the current working directory."""
    if dotenv_path:
        dotenv.load_dotenv(dotenv_path=dotenv_path, override=override)
    if cwd:
        dotenv.load_dotenv(dotenv_path=Path.cwd() / ".env", override=override)
    if not dotenv_path and not cwd:
        raise ValueError("Either dotenv_path must be provided or cwd must be True.")


__all__ = [
    # submodules
    "plugins",
    # agent
    "Agent",
    "History",
    # llm
    "ChatCompletion",
    "TextStream",
    "ReasoningStream",
    "MessageStream",
    "ChatCompletionStream",
    "ChatCompletionEvents",
    "LLMOptions",
    "ReasoningOptions",
    # models
    "DataContent",
    "FinishReason",
    "ToolChoice",
    "Usage",
    "ToolCallPart",
    "ToolResultPart",
    "ToolCall",
    "ToolCallResponse",
    "File",
    "UrlCitation",
    "Annotation",
    "UserConsentRequest",
    "MessageBase",
    "TextPart",
    "ReasoningPart",
    "MessagePartText",
    "MessagePartReasoning",
    "MessagePartFile",
    "MessagePartToolCall",
    "MessagePartToolResult",
    "MessagePart",
    "UserMessagePart",
    "AssistantMessagePart",
    "UserMessage",
    "ObjectType",
    "AssistantMessage",
    "ToolMessage",
    "Message",
    "Prompt",
    "ResponseFormatText",
    "ResponseFormatJson",
    "ResponseFormat",
    "StreamPartTextStart",
    "StreamPartTextDelta",
    "StreamPartTextEnd",
    "StreamPartReasoningStart",
    "StreamPartReasoningDelta",
    "StreamPartReasoningEnd",
    "StreamPart",
    # tools & plugins
    "Plugin",
    "PluginInitError",
    "Skills",
    "Skill",
    "magic",
    "tool",
    "ImageUrl",
    "Tool",
    "Tools",
    "ToolResult",
    "MCP",
    "MCPContext",
    # stream
    "StreamPartTurnStart",
    "StreamPartTurnEnd",
    # utils
    "init_logging",
    "load_dotenv",
]
