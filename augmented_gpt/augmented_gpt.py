from typing import (
    Callable,
    AsyncGenerator,
    List,
    Literal,
    TypeVar,
    Any,
    Generic,
    overload,
    TYPE_CHECKING,
)

from .message import *
from .tools import ToolInfo, ToolRegistry, Tools
from .history import History
import os

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .plugins import Plugin
    from .llm import GPTModel, GPTOptions

M = TypeVar("M", Message, MessageStream)


class ChatCompletion(Generic[M]):
    def __init__(self, agen: AsyncGenerator[M, None]) -> None:
        super().__init__()
        self.__agen = agen

    async def __anext__(self) -> M:
        return await self.__agen.__anext__()

    def __aiter__(self):
        return self


class AugmentedGPT:
    def support_tools(self) -> bool:
        return "vision" not in self.__backend.model

    def __init__(
        self,
        model: "GPTModel" = "gpt-4-turbo",
        tools: Optional[Tools] = None,
        gpt_options: Optional["GPTOptions"] = None,
        api_key: Optional[str] = None,
        instructions: Optional[str] = None,
        llm: Literal["openai"] = "openai",
        name: Optional[str] = None,
        description: Optional[str] = None,
        debug: bool = False,
    ):
        _api_key = api_key or os.environ.get("OPENAI_API_KEY")
        assert _api_key is not None, "Missing OPENAI_API_KEY"
        from .llm import LLMBackend, GPTOptions
        from .llm.openai import OpenAIBackend

        if llm == "openai":
            self.__backend: LLMBackend = OpenAIBackend(
                model=model,
                tools=ToolRegistry(self, tools),
                gpt_options=gpt_options or GPTOptions(),
                api_key=_api_key,
                instructions=instructions,
                debug=debug,
            )
        else:
            raise NotImplemented
        self.on_tool_start: Optional[Callable[[str, ToolInfo, Any], Any]] = None
        self.on_tool_end: Optional[Callable[[str, ToolInfo, Any, Any], Any]] = None
        self.name = name
        self.description = description

    def reset(self):
        self.__backend.reset()

    # @property
    # def openai_client(self):
    #     return self.__backend.client

    def get_plugin(self, name: str) -> "Plugin":
        return self.__backend.tools.get_plugin(name)

    @overload
    def chat_completion(
        self,
        messages: List[Message],
        stream: Literal[False] = False,
        context: Any = None,
    ) -> ChatCompletion[Message]: ...

    @overload
    def chat_completion(
        self, messages: List[Message], stream: Literal[True]
    ) -> ChatCompletion[MessageStream]: ...

    def chat_completion(
        self,
        messages: list[Message],
        stream: bool = False,
        context: Any = None,
    ) -> ChatCompletion[MessageStream] | ChatCompletion[Message]:
        if stream:
            return self.__backend.chat_completion(
                messages,
                stream=True,
                context=context,
            )
        else:
            return self.__backend.chat_completion(
                messages,
                stream=False,
                context=context,
            )

    def get_history(self) -> History:
        return self.__backend.get_history()

    def get_model(self) -> "GPTModel":
        return self.__backend.model

    def get_tools(self) -> ToolRegistry:
        return self.__backend.tools
