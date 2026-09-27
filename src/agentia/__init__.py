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
from .tools import ProviderTool, Tool, Tools, MCP, MCPContext, ToolResult
from .history import History
from .llm import LLMOptions, ReasoningOptions
from .llm.completion import Run
from .llm.stream import (
    TextStream,
    ReasoningStream,
    MessageStream,
    RunStream,
    RunEvents,
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
    "Run",
    "RunStream",
    "RunEvents",
    "TextStream",
    "ReasoningStream",
    "MessageStream",
    "LLMOptions",
    "ReasoningOptions",
    # models -- base
    "DataContent",
    "FinishReason",
    "ToolChoice",
    "Usage",
    "FunctionTool",
    "ProviderDefinedTool",
    "ToolSchema",
    "File",
    "UrlCitation",
    "Annotation",
    "UserConsentRequest",
    # models -- content parts
    "InputText",
    "InputImage",
    "InputFile",
    "InputAudio",
    "InputAudioData",
    "OutputText",
    "Refusal",
    "ReasoningText",
    "SummaryText",
    "InputContent",
    "OutputContent",
    "Content",
    "ReasoningContent",
    "input_content_from_file",
    # models -- items
    "ItemStatus",
    "MessageRole",
    "ObjectType",
    "MessageInput",
    "Message",
    "FunctionCall",
    "FunctionCallOutput",
    "Reasoning",
    "Compaction",
    "Item",
    "InputItem",
    "to_responses_input",
    # models -- options
    "ResponseFormatText",
    "ResponseFormatJson",
    "ResponseFormat",
    # models -- stream events
    "ResponseStatus",
    "ResponseInfo",
    "ResponseCreatedEvent",
    "ResponseInProgressEvent",
    "ResponseCompletedEvent",
    "ResponseIncompleteEvent",
    "ResponseFailedEvent",
    "ResponseOutputItemAddedEvent",
    "ResponseOutputItemDoneEvent",
    "ResponseContentPartAddedEvent",
    "ResponseContentPartDoneEvent",
    "ResponseOutputTextDeltaEvent",
    "ResponseOutputTextDoneEvent",
    "ResponseOutputTextAnnotationAddedEvent",
    "ResponseRefusalDeltaEvent",
    "ResponseRefusalDoneEvent",
    "ResponseReasoningTextDeltaEvent",
    "ResponseReasoningTextDoneEvent",
    "ResponseReasoningSummaryTextDeltaEvent",
    "ResponseReasoningSummaryTextDoneEvent",
    "ResponseReasoningSummaryPartAddedEvent",
    "ResponseReasoningSummaryPartDoneEvent",
    "ResponseFunctionCallArgumentsDeltaEvent",
    "ResponseFunctionCallArgumentsDoneEvent",
    "ErrorEvent",
    "RunStartedEvent",
    "RunCompletedEvent",
    "InputItemAddedEvent",
    "StreamEvent",
    # tools & plugins
    "Plugin",
    "PluginInitError",
    "Skills",
    "Skill",
    "magic",
    "tool",
    "ImageUrl",
    "ProviderTool",
    "Tool",
    "Tools",
    "ToolResult",
    "MCP",
    "MCPContext",
    # utils
    "init_logging",
    "load_dotenv",
]
