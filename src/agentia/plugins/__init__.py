from agentia.tools import Plugin, PluginInitError

from ..utils.decorators import tool

# Plugins
from .skills import Skills
from .system import System
from .web import Web

__all__ = [
    "Plugin",
    "PluginInitError",
    "Skills",
    "System",
    "Web",
    "tool",
]
