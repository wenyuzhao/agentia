import os
from ..utils.decorators import tool
from agentia.tools import Plugin, PluginInitError


__all__ = [
    "Plugin",
    "PluginInitError",
    "tool",
]

# Plugins
from .calc import Calculator
from .clock import Clock
from .code_runner import CodeRunner
from .skills import Skills
from .skill_learner import SkillLearner
from .memory import Memory

# Optional plugins
try:
    from .search import Search
    from .web import Web
except ImportError as e:
    pass
    # raise e
