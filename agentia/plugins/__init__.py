import abc
import os
import tomlkit.container
from ..decorators import tool
from ..message import Message
from typing import TYPE_CHECKING, Any, Callable, Self, Type
import tomlkit

if TYPE_CHECKING:
    from ..agent import Agent


class PluginInitError(RuntimeError):
    def __init__(self, plugin: str, original: Exception) -> None:
        self.plugin = plugin
        self.msg = str(original)
        self.original = original
        super().__init__(f"Plugin {plugin} failed to initialize: {self.msg}")


class ToolResult(Exception):
    __result: Any

    def __init__(self, result: Any):
        self.__result = result

    @property
    def result(self) -> Any:
        return self.__result


class Plugin(abc.ABC):
    NAME: str | None = None
    _BUILTIN_ID: str | None = None

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

    def __init__(self):
        self.config: dict[str, Any] = {}
        self.agent: "Agent"

    @classmethod
    def instantiate(cls, config: dict[str, Any]) -> Self:
        """
        Instantiate the plugin with the given config.
        This is called when the agent is created.
        """
        try:
            obj = cls(**config)
            obj.config = config
            return obj
        except Exception as e:
            raise PluginInitError(cls.id(), e) from e

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
        from .dalle import DallEPlugin
        from .vision import VisionPlugin
        from .web import WebPlugin
        from .knowledge_base import KnowledgeBasePlugin

        ALL_PLUGINS = {
            "calc": CalculatorPlugin,
            "clock": ClockPlugin,
            "code": CodePlugin,
            "memory": MemoryPlugin,
            "search": SearchPlugin,
            "dalle": DallEPlugin,
            "vision": VisionPlugin,
            "web": WebPlugin,
            "knowledge_base": KnowledgeBasePlugin,
        }
    except ImportError as e:
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
