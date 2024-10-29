import asyncio
import logging
import shutil
from typing import (
    Annotated,
    Callable,
    AsyncGenerator,
    Coroutine,
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
import rich
from slugify import slugify
import weakref
import uuid

from agentia import MSG_LOGGER
from agentia.retrieval import KnowledgeBase, VectorStore

from .message import *
from .history import History

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tools import ToolInfo, ToolRegistry, Tools
    from .plugins import Plugin
    from .llm import ModelOptions

M = TypeVar("M", AssistantMessage, MessageStream)

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


ToolCallEventListener = Callable[[ToolCallEvent], Any]
CommunicationEventListener = Callable[[CommunicationEvent], Any]

DEFAULT_MODEL = "openai/gpt-4o-mini"


@dataclass
class ChatCompletion(Generic[M]):
    def __init__(self, agent: "Agent", agen: AsyncGenerator[M, None]) -> None:
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

    async def dump(self):
        await self.__agent.init()

        def print_name_and_icon(name: str, icon: str | None):
            name_and_icon = f"[{icon} {name}]" if icon else f"[{name}]"
            rich.print(f"[bold blue]{name_and_icon}[/bold blue]")

        async for msg in self.__agen:
            if isinstance(msg, Message):
                print_name_and_icon(self.__agent.name, self.__agent.icon)
                print(msg.content)
            if isinstance(msg, MessageStream):
                name_printed = False
                outputed = False
                async for delta in msg:
                    if delta == "":
                        continue
                    if not name_printed:
                        print_name_and_icon(self.__agent.name, self.__agent.icon)
                        name_printed = True
                    outputed = True
                    print(delta, end="", flush=True)
                if outputed:
                    print()


UserConsentHandler = Callable[[str], bool | Coroutine[Any, Any, bool]]


class Agent:
    def __init__(
        self,
        name: str = "default",
        icon: str | None = None,
        description: str | None = None,
        model: Annotated[str | None, f"Default to {DEFAULT_MODEL}"] = None,
        tools: Optional["Tools"] = None,
        options: Optional["ModelOptions"] = None,
        api_key: str | None = None,
        instructions: str | None = None,
        debug: bool = False,
        colleagues: list["Agent"] | None = None,
        knowledge_base: KnowledgeBase | bool | str | None = None,
        persist_session: bool = False,
    ):
        from .llm import LLMBackend, ModelOptions
        from .tools import ToolRegistry

        # Init simple fields
        self.__is_initialized = False
        name = name.strip()
        if name == "":
            raise ValueError("Agent name cannot be empty.")
        self.name = name
        self.id = slugify(name.lower())
        self.session_id = self.id + "-" + str(uuid.uuid4())
        self.icon = icon
        self.log = MSG_LOGGER.getChild(self.id)
        if debug:
            self.log.setLevel(logging.DEBUG)
        model = model or DEFAULT_MODEL
        if ":" in model:
            provider = model.split(":")[0]
            model = model.split(":")[1]
        else:
            provider = "openrouter"
        self.description = description
        self.colleagues: dict[str, "Agent"] = {}
        self.__user_consent_handler: UserConsentHandler | None = None
        self.__on_tool_start: Callable[[ToolCallEvent], Any] | None = None
        self.__on_tool_end: Callable[[ToolCallEvent], Any] | None = None
        self.__on_communication_start: Callable[[CommunicationEvent], Any] | None = None
        self.__on_communication_end: Callable[[CommunicationEvent], Any] | None = None
        self.context: Any = None
        self.original_config: Any = None
        self.agent_data_folder = _get_global_cache_dir() / "agents" / f"{self.id}"
        self.session_data_folder = (
            _get_global_cache_dir() / "sessions" / f"{self.session_id}"
        )
        self.agent_data_folder.mkdir(parents=True, exist_ok=True)
        self.session_data_folder.mkdir(parents=True, exist_ok=True)
        self.persist_session = persist_session
        self.__tools = ToolRegistry(self, tools)
        self.__instructions = instructions
        # Init colleagues
        if colleagues is not None and len(colleagues) > 0:
            self.__init_cooperation(colleagues)
        # Init knowledge base (Step 1)
        if knowledge_base is True:
            self.knowledge_base = KnowledgeBase(
                self.session_data_folder / "knowledge-base"
            )
        elif isinstance(knowledge_base, KnowledgeBase):
            self.knowledge_base = knowledge_base
        elif isinstance(knowledge_base, str):
            self.knowledge_base = KnowledgeBase(knowledge_base)
        else:
            self.knowledge_base = None
        if self.knowledge_base is not None:
            self.__init_knowledge_base()
        # Init memory
        self.__init_memory()
        # Init history and backend
        self.__history = History(instructions=self.__instructions)
        if provider == "openai":
            from .llm.openai import OpenAIBackend

            self.__backend: LLMBackend = OpenAIBackend(
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

        if not persist_session:
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

    async def request_for_user_consent(self, message: str) -> bool:
        if self.__user_consent_handler is not None:
            result = self.__user_consent_handler(message)
            if asyncio.iscoroutine(result):
                return await result
            return result
        return True

    def on_user_consent(self, listener: UserConsentHandler):
        self.__user_consent_handler = listener
        return listener

    def on_tool_start(self, listener: Callable[[ToolCallEvent], Any]):
        self.__on_tool_start = listener
        return listener

    def on_tool_end(self, listener: Callable[[ToolCallEvent], Any]):
        self.__on_tool_end = listener
        return listener

    def on_commuication_start(self, listener: Callable[[CommunicationEvent], Any]):
        self.__on_communication_start = listener
        return listener

    def on_commuication_end(self, listener: Callable[[CommunicationEvent], Any]):
        self.__on_communication_end = listener
        return listener

    async def _emit_tool_call_event(self, event: ToolCallEvent):
        async def call_listener(listener: ToolCallEventListener):
            result = listener(event)
            if asyncio.iscoroutine(result):
                await result

        if event.result is None and self.__on_tool_start is not None:
            await call_listener(self.__on_tool_start)
        if event.result is not None and self.__on_tool_end is not None:
            await call_listener(self.__on_tool_end)

    async def _emit_communication_event(self, event: CommunicationEvent):
        async def call_listener(listener: CommunicationEventListener):
            result = listener(event)
            if asyncio.iscoroutine(result):
                await result

        if event.response is None and self.__on_communication_start is not None:
            await call_listener(self.__on_communication_start)
        if event.response is not None and self.__on_communication_end is not None:
            await call_listener(self.__on_communication_end)

    def __add_colleague(self, colleague: "Agent"):
        if colleague.name in self.colleagues:
            return
        self.colleagues[colleague.name] = colleague
        # Add a tool to dispatch a job to one colleague
        agent_names = [agent.name for agent in self.colleagues.values()]
        leader = self
        description = "Send a message or dispatch a job to a agent, and get the response from them. Note that the agent does not have any context expect what you explicitly told them, so give them the details as precise and as much as possible. Agents cannot contact each other, please coordinate the jobs and information between them properly by yourself when necessary. Here are a list of agents with their description:\n"
        for agent in self.colleagues.values():
            description += f" * {agent.name}: {agent.description}\n"

        from .decorators import tool

        @tool(name="_communiate", description=description)
        async def communiate(
            agent: Annotated[
                Annotated[str, agent_names],
                "The name of the agent to communicate with. This must be one of the provided colleague names.",
            ],
            message: Annotated[
                str, "The message to send to the agent, or the job details."
            ],
        ):
            self.log.info(f"COMMUNICATE {leader.name} -> {agent}: {repr(message)}")

            target = self.colleagues[agent]
            cid = uuid.uuid4().hex
            await self._emit_communication_event(
                CommunicationEvent(id=cid, parent=leader, child=target, message=message)
            )
            response = target.chat_completion(
                [
                    SystemMessage(
                        f"{leader.name} is directly talking to you right now. ({leader.name}: {leader.description})",
                    ),
                    UserMessage(message),
                ]
            )
            last_message = ""
            async for m in response:
                if isinstance(m, Message):
                    self.log.info(
                        f"RESPONSE {leader.name} <- {agent}: {repr(m.content)}"
                    )
                    # results.append(m.to_json())
                    last_message = m.content
                    await self._emit_communication_event(
                        CommunicationEvent(
                            id=cid,
                            parent=leader,
                            child=target,
                            message=message,
                            response=m.content,
                        )
                    )
            return last_message

        self.__tools._add_dispatch_tool(communiate)

    def __init_cooperation(self, colleagues: list["Agent"]):
        # Leader can dispatch jobs to colleagues
        for colleague in colleagues:
            self.__add_colleague(colleague)

    def __init_knowledge_base(self):
        if self.knowledge_base is None:
            return
        # Load global documents
        docs = self.agent_data_folder / "knowledge-base" / "docs"
        docs.mkdir(parents=True, exist_ok=True)
        global_vector_store = VectorStore(self.agent_data_folder / "knowledge-base")
        if len(global_vector_store.initial_files or []) > 0:
            self.knowledge_base.add_vector_store("global", global_vector_store)
            files = global_vector_store.initial_files or []
            if self.__instructions is not None:
                self.__instructions += f"\n\nFILES: {', '.join(files)}"
            else:
                self.__instructions = f"FILES: {', '.join(files)}"

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

    def __init_memory(self):
        from .plugins import MemoryPlugin

        mem_plugin = self.get_plugin("Memory")
        if mem_plugin is None or not isinstance(mem_plugin, MemoryPlugin):
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
        self, messages: Sequence[Message] | str, stream: Literal[False] = False
    ) -> ChatCompletion[AssistantMessage]: ...

    @overload
    def chat_completion(
        self, messages: Sequence[Message] | str, stream: Literal[True]
    ) -> ChatCompletion[MessageStream]: ...

    def chat_completion(
        self,
        messages: Sequence[Message] | str,
        stream: bool = False,
    ) -> ChatCompletion[MessageStream] | ChatCompletion[AssistantMessage]:
        if isinstance(messages, str):
            messages = [UserMessage(messages)]
        self.__load_files(messages)
        if stream:
            return self.__backend.chat_completion(messages, stream=True)
        else:
            return self.__backend.chat_completion(messages, stream=False)

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
        self.knowledge_base.add_documents(files)

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
