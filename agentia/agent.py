from datetime import datetime
import logging
from pathlib import Path
from typing import (
    Annotated,
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
import shelve
import rich
import rich.panel
from slugify import slugify
import weakref
import uuid
import os
import shortuuid

from agentia import LOGGER

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


def get_default_model() -> str:
    if path := os.environ.get("AGENTIA_DEFAULT_MODEL"):
        return path
    DEFAULT_MODEL_OPENROUTER = "openai/gpt-4o-mini"
    DEFAULT_MODEL_OPENAI = "gpt-4o-mini"
    if "OPENAI_BASE_URL" in os.environ:
        return "openai/gpt-4o-mini"
    else:
        return "gpt-4o-mini"


PluginType = TypeVar("PluginType", bound="Plugin")


class Agent:
    def __init__(
        self,
        # Agent ID
        id: str | None = None,
        # Identity and instructions
        name: str | None = None,
        icon: str | None = None,
        description: str | None = None,
        instructions: str | None = None,
        user: str | None = None,
        # Cooperation
        subagents: list["Agent"] | None = None,
        # Model and tools
        model: Annotated[str | None, f"Default to {get_default_model()}"] = None,
        options: Optional["ModelOptions"] = None,
        tools: Optional["Tools"] = None,
        api_key: str | None = None,
        # Session recovery
        persist: bool = False,
        session_id: str | None = None,
    ):
        from .tools import ToolRegistry
        from .utils.session import get_global_cache_dir
        from agentia.llm import create_llm_backend

        # Init simple fields
        self.__is_initialized = False
        if name is not None:
            name = name.strip()
            if name == "":
                raise ValueError("Agent name cannot be empty.")
        self.name = name
        self.id = slugify((id or shortuuid.uuid()[:16]).strip())
        self.id_is_random = id is None

        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        if persist:
            self.session_id = session_id or (
                self.id + "-" + timestamp + "-" + str(shortuuid.uuid()[:16])
            )
        else:
            if session_id:
                raise ValueError("Session ID cannot be set when persist is False.")
            self.session_id = (
                self.id + "-" + timestamp + "-" + str(shortuuid.uuid()[:16])
            )
        self.persist = persist
        self.icon = icon
        self.log = LOGGER.getChild(self.id)
        self.__init_logger()
        icon_and_name = ((icon or "") + " " + (name or "")).strip()
        self.log.info(
            f"Creating Agent: {self.id} {f'({icon_and_name})' if icon_and_name else ''}"
        )
        model = model or get_default_model()
        self.description = description
        self.subagents: dict[str, "Agent"] = {}
        self.context: Any = None
        self.config: Optional["Config"] = None
        self.config_path: Optional[Path] = None
        self.agent_data_folder = get_global_cache_dir() / "agents" / f"{self.id}"
        self.session_data_folder = (
            get_global_cache_dir() / "sessions" / f"{self.session_id}"
        )
        self.__user_consent = False
        self.__tools = ToolRegistry(self, tools)
        # Generate instructions
        self.__instructions = instructions
        if self.name and not self.description:
            self.__instructions = f"YOUR ARE {self.name}\n\n{self.__instructions or ''}"
        elif self.description and not self.name:
            self.__instructions = (
                f"YOUR DESCRIPTION: {self.description}\n\n{self.__instructions or ''}"
            )
        elif self.description and self.name:
            self.__instructions = f"YOUR ARE {self.name}, {self.description}\n\n{self.__instructions or ''}"
        if user:
            self.__instructions = f"{self.__instructions or ''}\n\nYOU ARE TALKING TO THE USER.\nMY INFO (THE USER): {user}"
        else:
            self.__instructions = (
                f"{self.__instructions or ''}\n\nYOU ARE TALKING TO THE USER."
            )

        # Init subagents
        if subagents is not None and len(subagents) > 0:
            self.__init_cooperation(subagents)
        # Init history and backend
        self.__history = History(instructions=self.__instructions)
        self.__backend = create_llm_backend(
            model=model,
            options=options,
            api_key=api_key,
            tools=self.__tools,
            history=self.__history,
        )

        weakref.finalize(self, Agent.__sweeper, self.session_id, self.persist)

    @staticmethod
    def __sweeper(session_id: str, persist: bool):
        from .utils.session import delete_session

        if not persist:
            delete_session(session_id)

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
        agent_path = (
            self.config_path.relative_to(Path.cwd()) if self.config_path else ""
        )
        header = f"[bold blue]Configuring plugins:[/bold blue] [blue]{self.id}[/blue] [dim italic]{agent_path}[/dim italic]"

        if Agent.is_cli():
            rich.print(rich.panel.Panel.fit(header))
        await self.__backend.tools.init(silent=not Agent.is_cli())
        if Agent.is_cli():
            rich.print()

    def __add_subagent(self, agent: "Agent"):
        if agent.name is None:
            raise ValueError("Agent name is required for cooperation.")
        if agent.description is None:
            raise ValueError("Agent description is required for cooperation.")
        if agent.id in self.subagents:
            return
        self.subagents[agent.id] = agent
        # Add a tool to dispatch a job to a subagent
        agent_ids = [agent.id for agent in self.subagents.values()]
        leader = self
        description = f"Send a message or dispatch a job to a person/subagent below as yourself ({leader.name}), and get the response from them. Note that the person does not have any context except what you explicitly told them, so give them the details as precise and as much as possible. They cannot contact each other, not the user, please coordinate everything between them properly by yourself. Here are a list of people with their description:\n"
        for agent in self.subagents.values():
            description += (
                f" * ID={agent.id} NAME={agent.name} -- {agent.description}\n"
            )

        from .tools import ToolResult, tool

        @tool(name="_communiate", description=description)
        async def communiate(
            id: Annotated[
                Annotated[str, agent_ids],
                "The id of the people to communicate with. This must be one of the provided ID.",
            ],
            message: Annotated[
                str,
                "The message to send, or the job details. You must send the message as yourself, not someone else.",
            ],
        ):
            self.log.debug(f"COMMUNICATE {leader.id} -> {id}: {repr(message)}")

            target = self.subagents[id]
            cid = uuid.uuid4().hex
            yield CommunicationEvent(
                id=cid, parent=leader.id, child=target.id, message=message
            )
            run = target.run(
                [
                    SystemMessage(
                        f"{leader.name} is directly talking to you ({target.name}) right now, not the user.\n\n{leader.name}'s INFO: {leader.description}\n\nYou are now talking and replying to {leader.name} not the user.",
                    ),
                    UserMessage(message),
                ]
            )
            last_message = ""
            async for m in run:
                if isinstance(
                    m, Union[UserMessage, SystemMessage, AssistantMessage, ToolMessage]
                ):
                    self.log.debug(f"RESPONSE {leader.id} <- {id}: {repr(m.content)}")
                    # results.append(m.to_json())
                    last_message = m.content
            yield CommunicationEvent(
                id=cid,
                parent=leader.id,
                child=target.id,
                message=message,
                response=last_message,
            )
            raise ToolResult(
                {
                    "response": last_message,
                    "description": f"This is the response from {target.name} and it is sent to you ({leader.name}), not the user.",
                }
            )

        self.__tools.add_tool(communiate)

    def __init_cooperation(self, subagents: list["Agent"]):
        if len(subagents) == 0:
            return
        if self.name is None:
            raise ValueError("Agent name is required for cooperation.")
        if self.description is None:
            raise ValueError("Agent description is required for cooperation.")
        # Leader can dispatch jobs to subagents
        for agent in subagents:
            self.__add_subagent(agent)

    def add_tool(self, f: Callable[..., Any]):
        """Add a tool to the agent"""
        self.__tools.add_tool(f)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__sweeper(self.session_id, self.persist)

    def open_configs_file(self):
        config_file = self.agent_data_folder / "config"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        return shelve.open(config_file)

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
        self.log.info("Agent Initialized")
        for c in self.subagents.values():
            await c.init()

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
            messages = [UserMessage(messages)]
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

    def all_agents(self) -> set["Agent"]:
        """Collect all active agents, including subagents"""
        agents = set()
        agents.add(self)
        for agent in self.subagents.values():
            agents.update(agent.all_agents())
        return agents

    @staticmethod
    def load_from_config(
        config: str | Path, persist: bool = False, session_id: str | None = None
    ):
        from .utils.config import load_agent_from_config

        return load_agent_from_config(config, persist, session_id)

    async def save(self):
        if not self.persist:
            return
        # History
        history_file = self.session_data_folder / "history"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        if self.history._first_conversation_just_finished():
            await self.summarise()
        self.history._save(history_file)
        # Subagent session IDs
        with shelve.open(history_file) as db:
            db["subagents"] = {
                k: v.session_id for k, v in self.subagents.items() if v.persist
            }
        # Save summary
        config_file = self.session_data_folder / "config"
        with shelve.open(config_file) as db:
            db["title"] = self.history.summary
            db["agent"] = self.id

    def anonymized(
        self, instructions: str | None = None, tools: Optional["Tools"] = None
    ) -> "Agent":
        """Create an agent with no name, no description, and by default no tools"""
        return Agent(
            instructions=instructions,
            model=self.__backend.get_default_model(),
            tools=tools,
        )

    async def summarise(self) -> str:
        """Summarise the history as a short title"""
        agent = self.anonymized(
            instructions="You need to summarise the following conversation as a short title. Just output the title, no other text, no quotes around it. The title should be short and precise, and it should be a single line. The title should not contain any other text."
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
