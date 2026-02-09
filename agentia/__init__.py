from __future__ import annotations

import logging

LOGGER = logging.getLogger("agentia")


def init_logging(level: logging._Level = logging.INFO):
    logging.basicConfig(
        level=level, format="[%(asctime)s][%(levelname)-8s][%(name)s] %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


from dotenv import load_dotenv

from . import utils

from .agent import Agent
from .spec import *
from .utils.decorators import magic, ImageUrl
from .utils.decorators import tool
from .plugins import Plugin, PluginInitError
from .plugins.skills import Skills, Skill
from . import llm, plugins
from .tools import ProviderTool, Tool, Tools, MCP, MCPContext
from .history import History
from .llm import LLMOptions
from .llm.completion import ChatCompletion
from .llm.stream import (
    TextStream,
    ReasoningStream,
    MessageStream,
    ChatCompletionStream,
    ChatCompletionEvents,
)

__all__ = [
    # submodules
    "plugins",
    "utils",
    "llm",
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
    # spec
    "ProviderOptions",
    "ProviderMetadata",
    "DataContent",
    "FinishReason",
    "ToolChoice",
    "Usage",
    "FunctionTool",
    "ProviderDefinedTool",
    "Tool",
    "ToolCall",
    "ToolResult",
    "File",
    "Annotation",
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
    "StreamPartStreamStart",
    "StreamPartStreamEnd",
    "StreamPart",
    # plugins
    "Plugin",
    "PluginInitError",
    "ToolResult",
    "Skills",
    "Skill",
    # utils
    "init_logging",
    "magic",
    "tool",
    "ImageUrl",
    "ProviderTool",
    "Tool",
    "Tools",
    "MCP",
    "MCPContext",
    "load_dotenv",
]
