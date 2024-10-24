from ..decorators import tool
from ..message import Message
import datetime
from typing import TYPE_CHECKING, Any
from agentia import tool

if TYPE_CHECKING:
    from ..agent import Agent


class Plugin:
    NAME: str | None

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

    def on_new_chat_message(self, msg: Message) -> Any: ...


class ClockPlugin(Plugin):
    @tool
    def get_current_time(self):
        """Get the current time in ISO format"""
        return datetime.datetime.now().isoformat()


from .calc import *
from .code import *
from .mstodo import *


ALL_PLUGINS = {
    "clock": ClockPlugin,
    "calc": CalculatorPlugin,
    "code": CodePlugin,
    "mstodo": MSToDoPlugin,
}
