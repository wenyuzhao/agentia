from __future__ import annotations

import logging

LOGGER = logging.getLogger("agentia")
MSG_LOGGER = LOGGER.getChild("message")

from .decorators import tool
from .message import (
    MessageStream,
    Message,
    UserMessage,
    SystemMessage,
    AssistantMessage,
    ToolMessage,
    Role,
    ToolCall,
    ContentPart,
    ContentPartImage,
    ContentPartText,
)
from .plugins import Plugin, ALL_PLUGINS
from . import plugins
from .agent import (
    Agent,
    ChatCompletion,
    ToolCallEventListener,
    UserConsentHandler,
    ToolCallEvent,
    Event,
)
from .llm import ModelOptions
from . import utils
from .tools import Tools


def init_logging(level: logging._Level = logging.INFO):
    logging.basicConfig(
        level=level, format="[%(asctime)s][%(levelname)-8s][%(name)s] %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


__all__ = [
    "utils",
    "tool",
    "Message",
    "UserMessage",
    "SystemMessage",
    "AssistantMessage",
    "ToolMessage",
    "MessageStream",
    "Role",
    "Plugin",
    "plugins",
    "Agent",
    "Tools",
    "ChatCompletion",
    "ToolCallEventListener",
    "UserConsentHandler",
    "ToolCallEvent",
    "ModelOptions",
    "ToolCall",
    "ContentPart",
    "ContentPartImage",
    "ContentPartText",
    "ALL_PLUGINS",
]
