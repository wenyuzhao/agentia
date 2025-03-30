from datetime import datetime
import logging
import shutil
from typing import (
    Annotated,
    AsyncGenerator,
    Literal,
    Optional,
    TypeVar,
    Any,
    Generic,
    overload,
    TYPE_CHECKING,
)
import shelve
import uuid
from slugify import slugify
import weakref
import uuid
import os
import shortuuid

from agentia import MSG_LOGGER

if TYPE_CHECKING:
    from agentia.knowledge_base import KnowledgeBase

from .message import *
from .history import History

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tools import ToolInfo, ToolRegistry, Tools
    from .plugins import Plugin
    from .llm import ModelOptions

_global_cache_dir = None


def _get_global_cache_dir() -> Path:
    global _global_cache_dir
    return _global_cache_dir or (Path.cwd() / ".cache")


@dataclass
class ToolCallEvent:
    agent: "Agent"
    tool: "ToolInfo"
    id: str
    function: FunctionCall
    result: Any | None = None


@dataclass
class CommunicationEvent:
    id: str
    parent: "Agent"
    child: "Agent"
    message: str
    response: str | None = None


@dataclass
class UserConsentEvent:
    id: str
    message: str
    response: bool | None = None


Event: TypeAlias = ToolCallEvent | CommunicationEvent | UserConsentEvent


DEFAULT_MODEL_OPENROUTER = "openai/gpt-4o-mini"
DEFAULT_MODEL_OPENAI = "gpt-4o-mini"
if "OPENAI_BASE_URL" in os.environ:
    DEFAULT_MODEL = DEFAULT_MODEL_OPENAI
else:
    DEFAULT_MODEL = DEFAULT_MODEL_OPENROUTER


M = TypeVar(
    "M",
    AssistantMessage,
    MessageStream,
    AssistantMessage | Event,
    MessageStream | Event,
)


@dataclass
class ChatCompletion(Generic[M]):
    def __init__(
        self,
        agent: "Agent",
        agen: AsyncGenerator[M, None],
    ) -> None:
        super().__init__()
        self.__agen = agen
        self.__agent = agent

    async def __anext__(self) -> M:
        await self.__agent.init()
        return await self.__agen.__anext__()

    def __aiter__(self):
        return self

    async def messages(self) -> AsyncGenerator[M, None]:
        await self.__agent.init()
        async for event in self:
            if isinstance(event, Message) or isinstance(event, MessageStream):
                yield event

    async def __await_impl(self) -> str:
        await self.__agent.init()
        last_message = ""
        async for msg in self.__agen:
            if isinstance(msg, Message):
                assert isinstance(msg.content, str)
                last_message = msg.content
            if isinstance(msg, MessageStream):
                last_message = ""
                async for delta in msg:
                    last_message += delta
        return last_message

    def __await__(self):
        return self.__await_impl().__await__()


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
        model: Annotated[str | None, f"Default to {DEFAULT_MODEL}"] = None,
        options: Optional["ModelOptions"] = None,
        tools: Optional["Tools"] = None,
        api_key: str | None = None,
        debug: bool = False,
    ):
        from .tools import ToolRegistry

        # Init simple fields
        self.__is_initialized = False
        if name is not None:
            name = name.strip()
            if name == "":
                raise ValueError("Agent name cannot be empty.")
        self.name = name
        self.id = slugify((id or shortuuid.uuid()[:16]).strip())

        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.session_id = self.id + "-" + timestamp + "-" + str(uuid.uuid4())
        self.icon = icon
        self.log = MSG_LOGGER.getChild(self.id)
        if debug:
            self.log.setLevel(logging.DEBUG)
        model = model or DEFAULT_MODEL
        self.description = description
        self.colleagues: dict[str, "Agent"] = {}
        self.context: Any = None
        self.original_config: Any = None
        self.agent_data_folder = _get_global_cache_dir() / "agents" / f"{self.id}"
        self.session_data_folder = (
            _get_global_cache_dir() / "sessions" / f"{self.session_id}"
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
        self.__init_backend(model, options, api_key)

        weakref.finalize(self, Agent.__sweeper, self.session_id)

    @staticmethod
    def __sweeper(session_id: str):
        session_dir = _get_global_cache_dir() / "sessions" / f"{session_id}"
        if session_dir.exists():
            shutil.rmtree(session_dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__sweeper(self.session_id)

    def open_configs_file(self):
        cache_file = self.agent_data_folder / "configs"
        return shelve.open(cache_file)

    @staticmethod
    def set_default_model(model: str):
        global DEFAULT_MODEL
        DEFAULT_MODEL = model

    @staticmethod
    def set_global_cache_dir(path: Path):
        global _global_cache_dir
        _global_cache_dir = path
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def init_logging(level: int = logging.INFO):
        """Initialize logging with a set of pre-defined rules."""
        from . import init_logging

        init_logging(level)

    async def init(self):
        if self.__is_initialized:
            return
        self.__is_initialized = True
        await self.__backend.tools.init()
        for c in self.colleagues.values():
            await c.init()

    def __init_backend(
        self, model: str, options: Optional["ModelOptions"], api_key: str | None
    ):
        from .llm import LLMBackend, ModelOptions

        if ":" in model:
            provider = model.split(":")[0]
            model = model.split(":")[1]
        elif "OPENAI_BASE_URL" in os.environ:
            provider = "openai"
        else:
            provider = "openrouter"
        assert provider in [
            "openai",
            "openrouter",
            "deepseek",
        ], f"Unknown provider: {provider}"
        if provider == "openai":
            from .llm.openai import OpenAIBackend

            self.__backend: LLMBackend = OpenAIBackend(
                model=model,
                tools=self.__tools,
                options=options or ModelOptions(),
                history=self.__history,
                api_key=api_key,
            )
        elif provider == "deepseek":
            from .llm.deepseek import DeepSeekBackend

            self.__backend: LLMBackend = DeepSeekBackend(
                model=model,
                tools=self.__tools,
                options=options or ModelOptions(),
                history=self.__history,
                api_key=api_key,
            )
        else:
            from .llm.openrouter import OpenRouterBackend

            self.__backend: LLMBackend = OpenRouterBackend(
                model=model,
                tools=self.__tools,
                options=options or ModelOptions(),
                history=self.__history,
                api_key=api_key,
            )

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
            self.log.info(f"COMMUNICATE {leader.id} -> {id}: {repr(message)}")

            target = self.colleagues[id]
            cid = uuid.uuid4().hex
            yield CommunicationEvent(
                id=cid, parent=leader, child=target, message=message
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
                    self.log.info(f"RESPONSE {leader.id} <- {id}: {repr(m.content)}")
                    # results.append(m.to_json())
                    last_message = m.content
            yield CommunicationEvent(
                id=cid,
                parent=leader,
                child=target,
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
                agent.log.info(f"FILE-SEARCH {query}")
            else:
                agent.log.info(f"FILE-SEARCH {query} ({filename})")
            assert agent.knowledge_base is not None
            response = await agent.knowledge_base.query(query, filename)
            return response

        self.__tools._add_file_search_tool(file_search)

        return knowledge_base

    def __init_memory(self):
        mem_plugin = self.get_plugin("Memory")
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

    def reset(self):
        self.history.reset()

    def get_plugin(self, name: str) -> Optional["Plugin"]:
        return self.__tools.get_plugin(name)

    @overload
    def chat_completion(
        self,
        messages: Sequence[Message] | str,
        *,
        stream: Literal[False] = False,
        events: Literal[False] = False,
    ) -> ChatCompletion[AssistantMessage]: ...

    @overload
    def chat_completion(
        self,
        messages: Sequence[Message] | str,
        *,
        stream: Literal[True],
        events: Literal[False] = False,
    ) -> ChatCompletion[MessageStream]: ...

    @overload
    def chat_completion(
        self,
        messages: Sequence[Message] | str,
        *,
        stream: Literal[False] = False,
        events: Literal[True],
    ) -> ChatCompletion[AssistantMessage | Event]: ...

    @overload
    def chat_completion(
        self,
        messages: Sequence[Message] | str,
        *,
        stream: Literal[True],
        events: Literal[True],
    ) -> ChatCompletion[MessageStream | Event]: ...

    def chat_completion(
        self,
        messages: Sequence[Message] | str,
        *,
        stream: bool = False,
        events: bool = False,
    ) -> (
        ChatCompletion[MessageStream]
        | ChatCompletion[AssistantMessage]
        | ChatCompletion[MessageStream | Event]
        | ChatCompletion[AssistantMessage | Event]
    ):
        if isinstance(messages, str):
            messages = [UserMessage(messages)]
        self.__load_files(messages)
        if stream and events:
            return self.__backend.chat_completion(messages, stream=True, events=True)
        elif stream:
            return self.__backend.chat_completion(messages, stream=True, events=False)
        elif events:
            return self.__backend.chat_completion(messages, stream=False, events=True)
        else:
            return self.__backend.chat_completion(messages, stream=False, events=False)

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

    @property
    def history(self) -> History:
        return self.__history

    @property
    def model(self) -> str:
        return self.__backend.model

    @property
    def tools(self) -> "ToolRegistry":
        return self.__backend.tools

    def all_agents(self) -> set["Agent"]:
        agents = set()
        agents.add(self)
        for agent in self.colleagues.values():
            agents.update(agent.all_agents())
        return agents

    @staticmethod
    def load_from_config(config: str | Path):
        from .utils.config import load_agent_from_config

        return load_agent_from_config(config)
