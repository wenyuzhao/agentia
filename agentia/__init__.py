from __future__ import annotations

import logging

LOGGER = logging.getLogger("agentia")


def init_logging(level: logging._Level = logging.INFO):
    logging.basicConfig(
        level=level, format="[%(asctime)s][%(levelname)-8s][%(name)s] %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


from . import message
from . import plugins
from . import utils

from .agent import Agent
from .message import (
    Message,
    Event,
    UserMessage,
    SystemMessage,
    AssistantMessage,
    UserConsentEvent,
    ToolCallEvent,
    CommunicationEvent,
)
from .chat_completion import ChatCompletion, MessageStream, ReasoningMessageStream
from .plugins import Plugin, tool

__all__ = [
    # submodules
    "message",
    "plugins",
    "utils",
    # agent
    "Agent",
    # message
    "Message",
    "Event",
    "UserMessage",
    "SystemMessage",
    "AssistantMessage",
    "UserConsentEvent",
    "ToolCallEvent",
    "CommunicationEvent",
    # chat_completion
    "ChatCompletion",
    "MessageStream",
    "ReasoningMessageStream",
    # plugins
    "Plugin",
    "tool",
    # utils
    "init_logging",
]
