from __future__ import annotations

import logging
from pathlib import Path

import dotenv

from . import plugins
from .agent import Agent
from .compaction import CompactionOptions
from .history import History, normalize_tool_pairs
from .llm import LLMOptions, ReasoningOptions, get_provider
from .llm.completion import Run
from .llm.loop import StepInfo, StopRun
from .llm.providers import LLMError, ModelRequest, Provider
from .llm.stream import (
    MessageStream,
    ReasoningStream,
    RunEvents,
    RunStream,
    TextStream,
)
from .mcp import MCPCatalog, MCPServerInfo, MCPToolInfo
from .models import *
from .plugins import Plugin, PluginInitError
from .plugins.skills import LocalSkillSource, Skill, Skills, SkillSource
from .tools import (
    MCP,
    BaseTool,
    MCPServers,
    MCPTools,
    ModelTool,
    ProviderTool,
    Suspend,
    Tool,
    ToolCall,
    ToolProvider,
    ToolResult,
    Tools,
)
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
    "BaseTool",
    "Compaction",
    "CompactionCompletedEvent",
    "CompactionOptions",
    "CompactionStartedEvent",
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
    "Instruction",
    "Item",
    "ItemStatus",
    "LLMError",
    "LLMOptions",
    "LocalSkillSource",
    "MCPCatalog",
    "MCPServerInfo",
    "MCPServers",
    "MCPToolInfo",
    "MCPTools",
    "Message",
    "MessageInput",
    "MessageRole",
    "MessageStream",
    "ModelRequest",
    "ModelTool",
    "ObjectType",
    "OutputContent",
    "OutputText",
    "Plugin",
    "PluginInitError",
    "Provider",
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
    "SkillSource",
    "Skills",
    "StepInfo",
    "StopRun",
    "StreamEvent",
    "SummaryText",
    "Suspend",
    "TextStream",
    "Tool",
    "ToolCall",
    "ToolChoice",
    "ToolProvider",
    "ToolResult",
    "ToolSchema",
    "Tools",
    "UrlCitation",
    "Usage",
    "UserConsentRequest",
    "get_provider",
    "init_logging",
    "input_content_from_file",
    "load_dotenv",
    "magic",
    "normalize_tool_pairs",
    "plugins",
    "to_responses_input",
    "tool",
]
