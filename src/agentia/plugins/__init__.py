from typing import TYPE_CHECKING

from agentia.tools import Plugin, PluginInitError

from ..utils.decorators import tool

# Plugins
from .skills import LocalSkillSource, Skill, Skills, SkillSource
from .system import System

if TYPE_CHECKING:
    from .web import Web


def __getattr__(name: str):
    # The Web plugin needs the optional `agentia[web]` dependencies.
    if name == "Web":
        from .web import Web

        return Web
    raise AttributeError(name)


__all__ = [
    "LocalSkillSource",
    "Plugin",
    "PluginInitError",
    "Skill",
    "SkillSource",
    "Skills",
    "System",
    "Web",
    "tool",
]
