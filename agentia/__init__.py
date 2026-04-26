from __future__ import annotations

import logging

from dotenv import load_dotenv

from .agent import Agent
from .models import *
from .utils.decorators import magic, ImageUrl
from .utils.decorators import tool
from .plugins import Plugin, PluginInitError
from .plugins.skills import Skills, Skill
from . import plugins
from .tools import ProviderTool, Tool, Tools, MCP, MCPContext, ToolResult
from .history import History
from .live import LiveOptions
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
    "ProviderOptions",
    "ProviderMetadata",
    "DataContent",
    "FinishReason",
    "ToolChoice",
    "Usage",
    "ToolCall",
    "ToolCallResponse",
    "File",
    "Annotation",
    "UserConsentRequest",
    "MessageBase",
    "SystemMessage",
    "MessagePartBase",
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
    "NonSystemMessage",
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
    "ProviderTool",
    "Tool",
    "Tools",
    "ToolResult",
    "MCP",
    "MCPContext",
    # live
    "LiveOptions",
    # stream (live)
    "StreamPartAudioStart",
    "StreamPartAudioDelta",
    "StreamPartAudioEnd",
    "StreamPartInputTranscriptionStart",
    "StreamPartInputTranscriptionDelta",
    "StreamPartInputTranscriptionEnd",
    "StreamPartOutputTranscriptionStart",
    "StreamPartOutputTranscriptionDelta",
    "StreamPartOutputTranscriptionEnd",
    "StreamPartTurnStart",
    "StreamPartTurnEnd",
    # utils
    "init_logging",
    "load_dotenv",
]
