from datetime import datetime
from io import BytesIO, StringIO
import logging
from pathlib import Path
from typing import (
    Annotated,
    Literal,
    Optional,
    Sequence,
    Type,
    Any,
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
    from agentia.knowledge_base import KnowledgeBase
    from agentia.chat_completion import ChatCompletion, MessageStream

from .message import *
from .history import History

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tools import ToolInfo, ToolRegistry, Tools
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
        colleagues: list["Agent"] | None = None,
        # Knowledge base
        knowledge_base: Union["KnowledgeBase", bool, Path, None] = None,
        # Model and tools
        model: Annotated[str | None, f"Default to {get_default_model()}"] = None,
        options: Optional["ModelOptions"] = None,
        tools: Optional["Tools"] = None,
        api_key: str | None = None,
        log_level: str | int | None = None,
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
        self.__init_logger(log_level)
        icon_and_name = ((icon or "") + " " + (name or "")).strip()
        self.log.info(
            f"Creating Agent: {self.id} {f'({icon_and_name})' if icon_and_name else ''}"
        )
        model = model or get_default_model()
        self.description = description
        self.colleagues: dict[str, "Agent"] = {}
        self.context: Any = None
        self.config: Optional["Config"] = None
        self.config_path: Optional[Path] = None
        self.agent_data_folder = get_global_cache_dir() / "agents" / f"{self.id}"
        self.session_data_folder = (
            get_global_cache_dir() / "sessions" / f"{self.session_id}"
        )
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
            self.__instructions = f"{self.__instructions or ''}\n\nYOU ARE NOW TALKING TO THE USER.\nMY INFO (THE USER): {user}"
        else:
            self.__instructions = (
                f"YOU ARE NOW TALKING TO THE USER.\n{self.__instructions or ''}"
            )
        # Init colleagues
        if colleagues is not None and len(colleagues) > 0:
            self.__init_cooperation(colleagues)
        # Init knowledge base (Step 1)
        self.knowledge_base: Optional["KnowledgeBase"] = None
        if knowledge_base is not False and knowledge_base is not None:
            self.knowledge_base = self.__init_knowledge_base(
                knowledge_base if knowledge_base is not True else None
            )
        # Init memory
        self.__init_memory()
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
    def history(self) -> History:
        return self.__history

    @property
    def model(self) -> str:
        return self.__backend.model

    @property
    def tools(self) -> "ToolRegistry":
        return self.__backend.tools

    def __init_logger(self, log_level: str | int | None):
        if log_level is None:
            if "LOG_LEVEL" in os.environ:
                log_level = os.environ["LOG_LEVEL"]
            elif Agent.is_server():
                log_level = logging.INFO
            else:
                log_level = logging.WARNING
        self.log.setLevel(log_level)

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

    def __add_colleague(self, colleague: "Agent"):
        if colleague.name is None:
            raise ValueError("Agent name is required for cooperation.")
        if colleague.description is None:
            raise ValueError("Agent description is required for cooperation.")
        if colleague.id in self.colleagues:
            return
        self.colleagues[colleague.id] = colleague
        # Add a tool to dispatch a job to one colleague
        agent_ids = [agent.id for agent in self.colleagues.values()]
        leader = self
        description = f"Send a message or dispatch a job to a person below as yourself ({leader.name}), and get the response from them. Note that the person does not have any context except what you explicitly told them, so give them the details as precise and as much as possible. They cannot contact each other, not the user, please coordinate everything between them properly by yourself. Here are a list of people with their description:\n"
        for agent in self.colleagues.values():
            description += (
                f" * ID={agent.id} NAME={agent.name} -- {agent.description}\n"
            )

        from .decorators import tool
        from .tools import ToolResult

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

            target = self.colleagues[id]
            cid = uuid.uuid4().hex
            yield CommunicationEvent(
                id=cid, parent=leader.id, child=target.id, message=message
            )
            response = target.chat_completion(
                [
                    SystemMessage(
                        f"{leader.name} is directly talking to you ({target.name}) right now, not the user.\n\n{leader.name}'s INFO: {leader.description}\n\nYou are now talking and replying to {leader.name} not the user.",
                    ),
                    UserMessage(message),
                ]
            )
            last_message = ""
            async for m in response:
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

        self.__tools._add_dispatch_tool(communiate)

    def __init_cooperation(self, colleagues: list["Agent"]):
        if len(colleagues) == 0:
            return
        if self.name is None:
            raise ValueError("Agent name is required for cooperation.")
        if self.description is None:
            raise ValueError("Agent description is required for cooperation.")
        # Leader can dispatch jobs to colleagues
        for colleague in colleagues:
            self.__add_colleague(colleague)

    def __init_knowledge_base(
        self, source: Union["KnowledgeBase", Path, None]
    ) -> "KnowledgeBase":
        from agentia.knowledge_base import KnowledgeBase

        # Get session store persist path
        session_store = self.session_data_folder / "knowledge-base"
        session_store.mkdir(parents=True, exist_ok=True)
        # Load knowledge base
        if isinstance(source, KnowledgeBase):
            # Load a pre-existing knowledge base
            knowledge_base = source
            knowledge_base.add_session_store(session_store)
        elif isinstance(source, Path):
            # Load knowledge base from a collection of documents
            persist_dir = self.agent_data_folder / "knowledge-base"
            knowledge_base = KnowledgeBase(
                global_store=persist_dir,
                global_docs=source,
                session_store=session_store,
            )
        else:
            # Create a new knowledge base or load a pre-existing one
            knowledge_base = KnowledgeBase(
                global_store=self.agent_data_folder / "knowledge-base",
                session_store=session_store,
            )
        # Update instructions
        global_vector_store = knowledge_base.vector_stores["global"]
        if len(global_vector_store.initial_files or []) > 0:
            files = global_vector_store.initial_files or []
            if self.__instructions is not None:
                self.__instructions += f"\n\nFILES: {', '.join(files)}"
            else:
                self.__instructions = f"FILES: {', '.join(files)}"

        # File search tool

        agent = self

        from .decorators import tool

        @tool(name="_file_search")
        async def file_search(
            query: Annotated[str, "The query to search for files"],
            filename: Annotated[
                str | None,
                "The optional filename of the file to search from. If not provided, search from all files.",
            ] = None,
        ):
            """Similarity-based search for related file segments in the knowledge base"""
            if filename is None:
                agent.log.debug(f"FILE-SEARCH {query}")
            else:
                agent.log.debug(f"FILE-SEARCH {query} ({filename})")
            assert agent.knowledge_base is not None
            response = await agent.knowledge_base.query(query, filename)
            return response

        self.__tools._add_file_search_tool(file_search)

        return knowledge_base

    def __init_memory(self):
        from .plugins.memory import MemoryPlugin

        mem_plugin = self.get_plugin(MemoryPlugin)
        if mem_plugin is None:
            return

        if not (self.agent_data_folder / "memory").exists():
            return
        content = (self.agent_data_folder / "memory").read_text().strip()

        if len(content) == 0:
            return

        if self.__instructions is None:
            self.__instructions = f"YOUR PREVIOUS MEMORY: \n{content}"
        else:
            self.__instructions += f"\n\nYOUR PREVIOUS MEMORY: \n{content}"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__sweeper(self.session_id, self.persist)

    def open_configs_file(self):
        config_file = self.agent_data_folder / "config"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        return shelve.open(config_file)

    @staticmethod
    def is_server() -> bool:
        v = os.environ.get("AGENTIA_SERVER", None)
        return v not in [None, "", "0", "false", "FALSE", "False"]

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
        for c in self.colleagues.values():
            await c.init()

    def reset(self):
        self.history.reset()

    def get_plugin(self, type: Type["Plugin"]) -> Optional["Plugin"]:
        return self.__tools.get_plugin(type)

    @overload
    def chat_completion(
        self,
        messages: Sequence[Message] | str,
        *,
        stream: Literal[False] = False,
        events: Literal[False] = False,
        response_format: Any | None = None,
    ) -> "ChatCompletion[AssistantMessage]": ...

    @overload
    def chat_completion(
        self,
        messages: Sequence[Message] | str,
        *,
        stream: Literal[True],
        events: Literal[False] = False,
        response_format: Any | None = None,
    ) -> "ChatCompletion[MessageStream]": ...

    @overload
    def chat_completion(
        self,
        messages: Sequence[Message] | str,
        *,
        stream: Literal[False] = False,
        events: Literal[True],
        response_format: Any | None = None,
    ) -> "ChatCompletion[AssistantMessage | Event]": ...

    @overload
    def chat_completion(
        self,
        messages: Sequence[Message] | str,
        *,
        stream: Literal[True],
        events: Literal[True],
        response_format: Any | None = None,
    ) -> "ChatCompletion[MessageStream | Event]": ...

    def chat_completion(
        self,
        messages: Sequence[Message] | str,
        *,
        stream: bool = False,
        events: bool = False,
        response_format: Any | None = None,
    ) -> Union[
        "ChatCompletion[MessageStream]",
        "ChatCompletion[AssistantMessage]",
        "ChatCompletion[MessageStream | Event]",
        "ChatCompletion[AssistantMessage | Event]",
    ]:
        if isinstance(messages, str):
            messages = [UserMessage(messages)]
        self.__load_files(messages)
        if stream and events:
            return self.__backend.chat_completion(
                messages, stream=True, events=True, response_format=response_format
            )
        elif stream:
            return self.__backend.chat_completion(
                messages, stream=True, events=False, response_format=response_format
            )
        elif events:
            return self.__backend.chat_completion(
                messages, stream=False, events=True, response_format=response_format
            )
        else:
            return self.__backend.chat_completion(
                messages, stream=False, events=False, response_format=response_format
            )

    def __load_files(self, messages: Sequence[Message]):
        old_messages = [m for m in messages]
        messages = []
        files: list[BytesIO] = []
        for m in old_messages:
            if isinstance(m, UserMessage) and m.files:
                filenames = []
                for file in m.files:
                    if isinstance(file, str) or isinstance(file, Path):
                        with open(file, "rb") as f:
                            f = BytesIO(f.read())
                            f.name = str(file)
                            files.append(f)
                            filenames.append(str(file))
                    elif isinstance(file, BytesIO):
                        files.append(file)
                        if not file.name:
                            raise ValueError(
                                "File name is required for BytesIO objects."
                            )
                        filenames.append(file.name)
                    elif isinstance(file, StringIO):
                        if not file.name:
                            raise ValueError(
                                "File name is required for StringIO objects."
                            )
                        filenames.append(file.name)
                        f = BytesIO(file.getvalue().encode())
                        f.name = f.name
                        files.append(f)
                messages.append(
                    SystemMessage(f"UPLOADED-FILES: {', '.join(filenames)}")
                )
            messages.append(m)
        if len(files) == 0:
            return
        if self.knowledge_base is None:
            raise ValueError("Knowledge base is disabled.")
        self.knowledge_base.add_temporary_documents(files)

    def all_agents(self) -> set["Agent"]:
        """Collect all active agents, including colleagues"""
        agents = set()
        agents.add(self)
        for agent in self.colleagues.values():
            agents.update(agent.all_agents())
        return agents

    @staticmethod
    def load_from_config(
        config: str | Path,
        persist: bool = False,
        session_id: str | None = None,
        log_level: str | int | None = None,
    ):
        from .utils.config import load_agent_from_config

        return load_agent_from_config(config, persist, session_id, log_level)

    async def save(self):
        if not self.persist:
            return
        # History
        history_file = self.session_data_folder / "history"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        if self.history._first_conversation_just_finished():
            await self.summarise()
        self.history._save(history_file)
        # Colleagues session IDs
        with shelve.open(history_file) as db:
            db["colleagues"] = {
                k: v.session_id for k, v in self.colleagues.items() if v.persist
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
        agent = Agent(
            instructions=instructions,
            model=self.__backend.get_default_model(),
            tools=tools,
            log_level=logging.WARNING,
        )
        return agent

    async def summarise(self) -> str:
        """Summarise the history as a short title"""
        agent = self.anonymized(
            instructions="You need to summarise the following conversation as a short title. Just output the title, no other text, no quotes around it. The title should be short and precise, and it should be a single line. The title should not contain any other text."
        )
        await agent.init()
        conversation = self.history.get_formatted_history()
        result = await agent.chat_completion(conversation)
        self.history.update_summary(result)
        return result


__all__ = ["Agent"]
