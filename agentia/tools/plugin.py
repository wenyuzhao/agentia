import abc
from typing import Any, Optional, Self, TYPE_CHECKING

if TYPE_CHECKING:
    from agentia.agent import Agent


class PluginInitError(RuntimeError):
    def __init__(self, plugin: str, original: Exception) -> None:
        self.plugin = plugin
        self.msg = str(original)
        self.original = original
        super().__init__(f"Plugin {plugin} failed to initialize: {self.msg}")


class Plugin(abc.ABC):
    NAME: str | None = None
    _BUILTIN_ID: str | None = None

    def get_instructions(self) -> str | None:
        """Get the instructions for using this plugin. This will be included in the agent's system prompt."""
        return None

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

    def __init__(self, *args: Any, **kwargs: Any):
        self.config: dict[str, Any] = {}
        self.agent: Optional["Agent"] = None

    @classmethod
    def instantiate(cls, config: dict[str, Any]) -> Self:
        """Instantiate the plugin with the given config."""
        try:
            obj = cls(**(config or {}))
            obj.config = config
            return obj
        except Exception as e:
            raise PluginInitError(cls.id(), e) from e

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
    def validate_config(cls, config: dict[str, Any]): ...
