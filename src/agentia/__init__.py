from __future__ import annotations

import logging
from pathlib import Path

import dotenv

from . import plugins
from .agent import Agent
from .history import History
from .llm import LLMOptions, ReasoningOptions
from .llm.completion import Run
from .llm.stream import (
    MessageStream,
    ReasoningStream,
    RunEvents,
    RunStream,
    TextStream,
)
from .models import *
from .plugins import Plugin, PluginInitError
from .plugins.skills import Skill, Skills
from .tools import MCP, MCPContext, ProviderTool, Tool, ToolResult, Tools
from .utils.decorators import ImageUrl, magic, tool

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
    "MCP",
    "Agent",
    "Annotation",
    "Compaction",
    "Content",
    "DataContent",
    "ErrorEvent",
    "File",
    "FinishReason",
    "FunctionCall",
    "FunctionCallOutput",
    "FunctionTool",
    "History",
    "ImageUrl",
    "InputAudio",
    "InputAudioData",
    "InputContent",
    "InputFile",
    "InputImage",
    "InputItem",
    "InputItemAddedEvent",
    "InputText",
    "Item",
    "ItemStatus",
    "LLMOptions",
    "MCPContext",
    "Message",
    "MessageInput",
    "MessageRole",
    "MessageStream",
    "ObjectType",
    "OutputContent",
    "OutputText",
    "Plugin",
    "PluginInitError",
    "ProviderDefinedTool",
    "ProviderTool",
    "Reasoning",
    "ReasoningContent",
    "ReasoningOptions",
    "ReasoningStream",
    "ReasoningText",
    "Refusal",
    "ResponseCompletedEvent",
    "ResponseContentPartAddedEvent",
    "ResponseContentPartDoneEvent",
    "ResponseCreatedEvent",
    "ResponseFailedEvent",
    "ResponseFormat",
    "ResponseFormatJson",
    "ResponseFormatText",
    "ResponseFunctionCallArgumentsDeltaEvent",
    "ResponseFunctionCallArgumentsDoneEvent",
    "ResponseInProgressEvent",
    "ResponseIncompleteEvent",
    "ResponseInfo",
    "ResponseOutputItemAddedEvent",
    "ResponseOutputItemDoneEvent",
    "ResponseOutputTextAnnotationAddedEvent",
    "ResponseOutputTextDeltaEvent",
    "ResponseOutputTextDoneEvent",
    "ResponseReasoningSummaryPartAddedEvent",
    "ResponseReasoningSummaryPartDoneEvent",
    "ResponseReasoningSummaryTextDeltaEvent",
    "ResponseReasoningSummaryTextDoneEvent",
    "ResponseReasoningTextDeltaEvent",
    "ResponseReasoningTextDoneEvent",
    "ResponseRefusalDeltaEvent",
    "ResponseRefusalDoneEvent",
    "ResponseStatus",
    "Run",
    "RunCompletedEvent",
    "RunEvents",
    "RunStartedEvent",
    "RunStream",
    "Skill",
    "Skills",
    "StreamEvent",
    "SummaryText",
    "TextStream",
    "Tool",
    "ToolChoice",
    "ToolResult",
    "ToolSchema",
    "Tools",
    "UrlCitation",
    "Usage",
    "UserConsentRequest",
    "init_logging",
    "input_content_from_file",
    "load_dotenv",
    "magic",
    "plugins",
    "to_responses_input",
    "tool",
]
