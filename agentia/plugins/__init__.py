from ..decorators import tool
from ..message import Message
from typing import TYPE_CHECKING, Any, Callable, Type

if TYPE_CHECKING:
    from ..agent import Agent


class ToolResult(BaseException):
    __result: Any

    def __init__(self, result: Any):
        self.__result = result

    @property
    def result(self) -> Any:
        return self.__result


class Plugin:
    NAME: str | None
    _BUILTIN_ID: str | None = None

    @property
    def id(self) -> str:
        if self._BUILTIN_ID:
            return self._BUILTIN_ID
        return self.name.lower()

    def __init__(self, config: Any = None):
        if hasattr(self, "NAME") and self.NAME:
            self.name = self.NAME.strip()
        else:
            self.name = self.__class__.__name__
            if self.name.endswith("Plugin"):
                self.name = self.name[:-6]
        self.cache_key = f"plugins.{self.name}".lower()
        self.config = config
        self.agent: "Agent"

    def _register(self, agent: "Agent"):
        self.agent = agent
        self.log = self.agent.log.getChild(self.name)

    async def init(self):
        pass

    @classmethod
    def validate_config(cls, config: dict[str, Any]): ...


def __import_plugins() -> dict[str, Type[Plugin]]:
    try:
        from . import (
            calc,
            clock,
            code,
            memory,
            mstodo,
            search,
            dalle,
            vision,
            web,
        )

        return {
            "calc": calc.CalculatorPlugin,
            "clock": clock.ClockPlugin,
            "code": code.CodePlugin,
            "memory": memory.MemoryPlugin,
            "mstodo": mstodo.MSToDoPlugin,
            "search": search.SearchPlugin,
            "dalle": dalle.DallEPlugin,
            "vision": vision.VisionPlugin,
            "web": web.WebPlugin,
        }
    except ImportError as e:
        # raise e
        return {}


ALL_PLUGINS = __import_plugins()

for name, cls in ALL_PLUGINS.items():
    cls._BUILTIN_ID = name


def register_plugin(name: str) -> Callable[[Type[Plugin]], Type[Plugin]]:
    assert name not in ALL_PLUGINS, f"Plugin {name} already registered"

    def wrapper(cls: Type[Plugin]):
        ALL_PLUGINS[name] = cls
        cls._BUILTIN_ID = name
        return cls

    return wrapper
