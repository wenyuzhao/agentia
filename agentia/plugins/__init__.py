import os
import tomlkit.container
from ..decorators import tool
from ..message import Message
from typing import TYPE_CHECKING, Any, Callable, Type
import tomlkit

if TYPE_CHECKING:
    from ..agent import Agent


class PluginInitError(RuntimeError):
    def __init__(self, plugin: str, original: BaseException) -> None:
        self.plugin = plugin
        self.msg = str(original)
        self.original = original
        super().__init__(f"Plugin {plugin} failed to initialize: {self.msg}")


class ToolResult(BaseException):
    __result: Any

    def __init__(self, result: Any):
        self.__result = result

    @property
    def result(self) -> Any:
        return self.__result


def _streamlit_enabled():
    try:
        import streamlit

        return True
    except ImportError:
        return False


class Plugin:
    STREAMLIT_ENABLED: bool = _streamlit_enabled()

    NAME: str | None = None
    _BUILTIN_ID: str | None = None

    @staticmethod
    def is_server() -> bool:
        from ..agent import Agent

        return Agent.is_server()

    @classmethod
    def name(cls) -> str:
        if cls.NAME:
            return cls.NAME
        name = cls.__name__
        if name.endswith("Plugin"):
            name = name[:-6]
        return name

    @classmethod
    def id(cls) -> str:
        if cls._BUILTIN_ID:
            return cls._BUILTIN_ID
        return cls.name().lower()

    @classmethod
    def cache_key(cls, k: str | None = None) -> str:
        key = f"plugins.{cls.id()}".lower()
        if k:
            while k.startswith("."):
                k = k[1:]
            key += f".{k}"
        return key

    def __init__(self, config: Any = None):
        self.config = config or {}
        self.agent: "Agent"

    def _register(self, agent: "Agent"):
        self.agent = agent
        self.log = self.agent.log.getChild(self.id())

    async def init(self):
        """
        Initialize the plugin before running the agent.
        This may involve creating API clients, login, and verify auth tokens.

        For OAuth:
            Login steps here can only access to CLI inputs.
            You may also want to override `__options__` so that user can login on the dashboard site.
        """
        pass

    @classmethod
    def __options__(cls, agent: str, config: tomlkit.container.Container):
        """
        Web UI for logging on the user and configuring the plugin.
        Any modifications to the `configs` parameter will be saved to the config file.

        For OAuth:
            Please call `self.agent.open_configs_file()` and save your oauth tokens there. Same for other secrets.
        """
        pass

    @classmethod
    def validate_config(cls, config: dict[str, Any]): ...


ALL_PLUGINS: dict[str, type[Plugin]] = {}

try:
    from .calc import CalculatorPlugin
    from .clock import ClockPlugin
    from .code import CodePlugin
    from .memory import MemoryPlugin
    from .mstodo import MSToDoPlugin
    from .search import SearchPlugin
    from .dalle import DallEPlugin
    from .vision import VisionPlugin
    from .web import WebPlugin
    from .gmail import GmailPlugin

    ALL_PLUGINS = {
        "calc": CalculatorPlugin,
        "clock": ClockPlugin,
        "code": CodePlugin,
        "memory": MemoryPlugin,
        "mstodo": MSToDoPlugin,
        "search": SearchPlugin,
        "dalle": DallEPlugin,
        "vision": VisionPlugin,
        "web": WebPlugin,
        "gmail": GmailPlugin,
    }
except ImportError as e:
    # raise e from e
    pass


for name, cls in ALL_PLUGINS.items():
    cls._BUILTIN_ID = name


def register_plugin(name: str) -> Callable[[Type[Plugin]], Type[Plugin]]:
    assert name not in ALL_PLUGINS, f"Plugin {name} already registered"

    def wrapper(cls: Type[Plugin]):
        ALL_PLUGINS[name] = cls
        cls._BUILTIN_ID = name
        return cls

    return wrapper
