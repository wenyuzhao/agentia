import os
from typing import Callable, Type
from ..utils.decorators import tool
from agentia.tools import Plugin, PluginInitError


ALL_PLUGINS: dict[str, type[Plugin]] = {}


__all__ = [
    "Plugin",
    "PluginInitError",
    "ALL_PLUGINS",
    "tool",
]

if os.environ.get("AGENTIA_DISABLE_PLUGINS", "").lower() not in [
    "1",
    "true",
    "yes",
    "y",
]:
    try:
        from .calc import CalculatorPlugin
        from .clock import ClockPlugin
        from .code import CodePlugin
        from .memory import MemoryPlugin
        from .search import SearchPlugin
        from .web import WebPlugin

        ALL_PLUGINS = {
            "calc": CalculatorPlugin,
            "clock": ClockPlugin,
            "code": CodePlugin,
            "memory": MemoryPlugin,
            "search": SearchPlugin,
            "web": WebPlugin,
        }
    except ImportError as e:
        print("Failed to import built-in plugins:", e)
        pass
        # raise e


for name, cls in ALL_PLUGINS.items():
    cls._BUILTIN_ID = name


def register_plugin(
    name: str, overwrite: bool = False
) -> Callable[[Type[Plugin]], Type[Plugin]]:
    if not overwrite:
        assert name not in ALL_PLUGINS, f"Plugin {name} already registered"

    def wrapper(cls: Type[Plugin]):
        ALL_PLUGINS[name] = cls
        cls._BUILTIN_ID = name
        return cls

    return wrapper
