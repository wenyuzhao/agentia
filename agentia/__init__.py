from __future__ import annotations

import logging

LOGGER = logging.getLogger("agentia")


def init_logging(level: logging._Level = logging.INFO):
    logging.basicConfig(
        level=level, format="[%(asctime)s][%(levelname)-8s][%(name)s] %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


from . import utils

from .agent import Agent
from .spec import (
    Message,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    MessagePart,
    MessagePartText,
    MessagePartFile,
    MessagePartReasoning,
    MessagePartToolCall,
    MessagePartToolResult,
    StreamPart,
    ToolCall,
    ToolResult,
)
from .utils.decorators import magic, ImageUrl
from .utils.decorators import tool
from .plugins import Plugin, PluginInitError, register_plugin
from . import llm
from . import plugins
from .llm.tools import ProviderTool, Tool, Tools, MCPServer

__all__ = [
    # submodules
    "plugins",
    "utils",
    "llm",
    # agent
    "Agent",
    # message
    "Message",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "MessagePart",
    "MessagePartText",
    "MessagePartFile",
    "MessagePartReasoning",
    "MessagePartToolCall",
    "MessagePartToolResult",
    "StreamPart",
    "ToolCall",
    "ToolResult",
    # plugins
    "Plugin",
    "PluginInitError",
    "ToolResult",
    "register_plugin",
    # utils
    "init_logging",
    "magic",
    "tool",
    "ImageUrl",
    "ProviderTool",
    "Tool",
    "Tools",
    "MCPServer",
]
