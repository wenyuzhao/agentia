import os
from ..utils.decorators import tool
from agentia.tools import Plugin, PluginInitError


__all__ = [
    "Plugin",
    "PluginInitError",
    "tool",
]

if os.environ.get("AGENTIA_DISABLE_PLUGINS", "").lower() not in [
    "1",
    "true",
    "yes",
    "y",
]:
    try:
        from .calc import Calculator
        from .clock import Clock
        from .code_runner import CodeRunner
        from .memory import Memory
        from .search import Search
        from .web import Web
        from .skills import Skills
        from .skill_learner import SkillLearner

    except ImportError as e:
        pass
        # raise e
