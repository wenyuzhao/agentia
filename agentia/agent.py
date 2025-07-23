from datetime import datetime
import logging
from pathlib import Path
from typing import (
    Callable,
    Literal,
    Optional,
    Sequence,
    Type,
    Any,
    TypeVar,
    Union,
    overload,
    TYPE_CHECKING,
)
import rich
import rich.panel
from slugify import slugify
import os
import shortuuid

from agentia import LOGGER
from agentia.mcp import MCPServer

if TYPE_CHECKING:
    from agentia.utils.config import Config
    from agentia.run import Run, MessageStream

from .message import *
from .history import History

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tools import ToolRegistry, Tools
    from .plugins import Plugin
    from .llm import ModelOptions


PluginType = TypeVar("PluginType", bound="Plugin")


class Agent:
    def __init__(
        self,
        # Agent ID, by default the agent will be assigned a random ID
        id: str | None = None,
        # instructions
        instructions: str | None = None,
        # Model and tools
        model: str | None = None,
        options: Optional["ModelOptions"] = None,
        tools: Optional["Tools"] = None,
        api_key: str | None = None,
    ):
        from .tools import ToolRegistry
        from agentia.llm import create_llm_backend, get_default_model

        # Init simple fields
        self.__is_initialized = False
        self.id = slugify((id or shortuuid.uuid()[:16]).strip())

        self.log = LOGGER.getChild(self.id)
        self.__init_logger()
        self.log.info(f"Creating Agent: {self.id}")
        model = model or get_default_model()
        self.context: dict[str, Any] = {}
        self.__user_consent = False
        self.__tools = ToolRegistry(self, tools)
        # Generate instructions
        self.__instructions = instructions
        # Init history and backend
        self.__history = History(instructions=self.__instructions)
        self.__backend = create_llm_backend(
            model=model,
            options=options,
            api_key=api_key,
            tools=self.__tools,
            history=self.__history,
        )

    @property
    def user_consent_enabled(self) -> bool:
        return self.__user_consent

    @property
    def history(self) -> History:
        return self.__history

    @property
    def model(self) -> str:
        return self.__backend.model

    @property
    def tools(self) -> "ToolRegistry":
        return self.__backend.tools

    def __init_logger(self):
        self.log.setLevel(os.environ.get("LOG_LEVEL", logging.WARNING))

    async def __init_plugins(self):
        header = f"[bold blue]Configuring plugins:[/bold blue] [blue]{self.id}[/blue]"

        if Agent.is_cli():
            rich.print(rich.panel.Panel.fit(header))
        await self.__backend.tools.init(silent=not Agent.is_cli())
        if Agent.is_cli():
            rich.print()

    def add_tools(self, *tools: Callable[..., Any] | MCPServer | "Plugin") -> None:
        """Add a tool to the agent"""
        for tool in tools:
            self.__tools.add(tool)

    @staticmethod
    def is_cli() -> bool:
        v = os.environ.get("AGENTIA_CLI", None)
        return v not in [None, "", "0", "false", "FALSE", "False"]

    async def init(self):
        if self.__is_initialized:
            return
        self.__is_initialized = True
        self.log.info("Agent Initializing ...")
        await self.__init_plugins()

    def reset(self):
        self.history.reset()

    def get_plugin(self, type: Type[PluginType]) -> Optional[PluginType]:
        p = self.__tools.get_plugin(type)
        if p is None:
            return None
        if not isinstance(p, type):
            return None
        return p

    @overload
    def run(
        self,
        messages: Sequence[Message] | UserMessage | str,
        *,
        stream: Literal[False] = False,
        events: Literal[False] = False,
        response_format: Any | None = None,
    ) -> "Run[AssistantMessage]": ...

    @overload
    def run(
        self,
        messages: Sequence[Message] | UserMessage | str,
        *,
        stream: Literal[True],
        events: Literal[False] = False,
        response_format: Any | None = None,
    ) -> "Run[MessageStream]": ...

    @overload
    def run(
        self,
        messages: Sequence[Message] | UserMessage | str,
        *,
        stream: Literal[False] = False,
        events: Literal[True],
        response_format: Any | None = None,
    ) -> "Run[AssistantMessage | Event]": ...

    @overload
    def run(
        self,
        messages: Sequence[Message] | UserMessage | str,
        *,
        stream: Literal[True],
        events: Literal[True],
        response_format: Any | None = None,
    ) -> "Run[MessageStream | Event]": ...

    def run(
        self,
        messages: Sequence[Message] | UserMessage | str,
        *,
        stream: bool = False,
        events: bool = False,
        response_format: Any | None = None,
    ) -> Union[
        "Run[MessageStream]",
        "Run[AssistantMessage]",
        "Run[MessageStream | Event]",
        "Run[AssistantMessage | Event]",
    ]:
        if isinstance(messages, str):
            messages = [UserMessage(content=messages)]
        elif isinstance(messages, UserMessage):
            messages = [messages]
        if stream and events:
            return self.__backend.run(
                messages, stream=True, events=True, response_format=response_format
            )
        elif stream:
            return self.__backend.run(
                messages, stream=True, events=False, response_format=response_format
            )
        elif events:
            return self.__backend.run(
                messages, stream=False, events=True, response_format=response_format
            )
        else:
            return self.__backend.run(
                messages, stream=False, events=False, response_format=response_format
            )

    async def summarise(self) -> str:
        """Summarise the history as a short title"""
        agent = Agent(
            instructions="You need to summarise the following conversation as a short title. Just output the title, no other text, no quotes around it. The title should be short and precise, and it should be a single line. The title should not contain any other text.",
            model=self.__backend.get_default_model(),
        )
        await agent.init()
        conversation = self.history.get_formatted_history()
        result = await agent.run(conversation)
        self.history.update_summary(result.content or "")
        return result.content or ""

    def enable_user_consent(self):
        """
        Require user consent for all important actions (e.g. executing code)

        This only serves a hint to the plugins that they should ask for user consent.
        Plugins may either ignore this or ask for user consent even if this is not set.
        """
        self.__user_consent = True


__all__ = ["Agent"]
