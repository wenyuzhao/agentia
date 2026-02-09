import os
import re
from typing import (
    TYPE_CHECKING,
    AsyncGenerator,
    Literal,
    Optional,
    Sequence,
    TypedDict,
    overload,
)
import httpx
from pydantic import AnyUrl

from agentia import spec
from agentia.llm.completion import ChatCompletion
from agentia.llm.stream import ChatCompletionEvents, ChatCompletionStream
from agentia.spec.chat import ObjectType
from agentia.spec.stream import *
from agentia.tools.tools import ToolSet, Tool
from dataclasses import dataclass

if TYPE_CHECKING:
    from agentia.agent import Agent
    from agentia.llm.providers import Provider
    from agentia.tools.mcp import MCPContext


class LLMOptionsDict(TypedDict, total=False):
    max_output_tokens: int | None
    temperature: float | None
    stop_sequences: Sequence[str] | None
    top_p: float | None
    top_k: int | None
    presence_penalty: float | None
    frequency_penalty: float | None
    seed: int | None
    tool_choice: spec.ToolChoice | None
    provider_options: spec.ProviderOptions | None
    response_format: spec.ResponseFormat | None
    tools: Sequence[Tool] | ToolSet | None
    parallel_tool_calls: bool | None


@dataclass
class LLMOptions:
    max_output_tokens: int | None = None
    temperature: float | None = None
    stop_sequences: Sequence[str] | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    tool_choice: spec.ToolChoice | None = None
    provider_options: spec.ProviderOptions | None = None
    response_format: spec.ResponseFormat | None = None
    tools: Sequence[Tool] | ToolSet | None = None
    parallel_tool_calls: bool | None = True


type LLMOptionsUnion = LLMOptions | LLMOptionsDict


def get_provider(selector: str) -> "Provider":
    DEFAULT_PROVIDER = os.environ.get("AGENTIA_PROVIDER", "openrouter")
    # if has no scheme, assume gateway
    if re.match(r"^\w+:", selector) is None:
        provider = DEFAULT_PROVIDER
        model = selector
    else:
        uri = AnyUrl(selector)
        provider = uri.scheme
        model = (uri.host or "") + (uri.path or "")
    model = model.strip("/")
    match provider:
        case "openai":
            from .providers.openai import OpenAI

            return OpenAI(model=model)
        case "gateway":
            from .providers.gateway import Gateway

            return Gateway(model=model)
        case "openrouter":
            from .providers.openrouter import OpenRouter

            return OpenRouter(model=model)
        case "qwen":
            from .providers.qwen import Qwen

            return Qwen(model=model)
        case "chutes":
            from .providers.chutes import Chutes

            return Chutes(model=model)
        case "fireworks":
            from .providers.fireworks import Fireworks

            return Fireworks(model=model)
        case "ollama":
            from .providers.ollama import Ollama

            return Ollama(model=model)
        case _:
            raise ValueError(f"Unknown provider: {provider}")


class LLM:
    def __init__(self, model: str) -> None:
        self._provider = get_provider(model)
        self._provider.llm = self
        self._agent: Optional["Agent"] = None
        self._model = model
        self._active_tools: ToolSet | None = None
        self._mcp_context: Optional["MCPContext"] = None

    @property
    def model(self) -> str:
        return self._model

    def __prepare_messages(
        self, prompt: str | spec.Message | Sequence[spec.Message]
    ) -> list[spec.Message]:
        messages: list[spec.Message] = []
        if isinstance(prompt, str):
            messages.append(
                spec.UserMessage(content=[spec.MessagePartText(text=prompt)])
            )
        elif isinstance(prompt, (list, Sequence)):
            messages.extend(prompt)
        else:
            messages.append(prompt)
        return messages

    async def __process_tool_calls(
        self, tool_calls: list[spec.ToolCall], tools: ToolSet | None, parallel: bool
    ) -> tuple[spec.ToolMessage, list[spec.ToolResult], spec.Message | None]:
        assert tools is not None, "No tools provided"
        tool_msg, tool_results, files = await tools.run(self, tool_calls, parallel)
        if files:
            msg = spec.UserMessage(
                content=[
                    spec.MessagePartText(text="[[TOOL_OUTPUT_FILES]]"),
                    *(
                        spec.MessagePartFile(
                            filename=f.id,
                            media_type=f.media_type,
                            data=f.data,
                        )
                        for f in files
                    ),
                ]
            )
        else:
            msg = None
        return tool_msg, tool_results, msg

    async def __init_tools(self, options: LLMOptions) -> ToolSet:
        if not options.tools:
            tools = ToolSet([])
        elif isinstance(options.tools, ToolSet):
            tools = options.tools
        else:
            tools = ToolSet(options.tools)
        await tools.init(self, self._agent)
        self._active_tools = tools
        return tools

    async def generate_object[T: ObjectType](
        self,
        prompt: str | spec.Message | Sequence[spec.Message],
        type: type[T],
        options: LLMOptionsUnion | None = None,
    ) -> T:
        msgs = await self._generate_object_impl(prompt, type, options)
        assert isinstance(msgs[-1], spec.AssistantMessage)
        return msgs[-1].parse(type)

    async def _generate_object_impl[T: ObjectType](
        self,
        prompt: str | spec.Message | Sequence[spec.Message],
        return_type: type[T],
        options: LLMOptionsUnion | None = None,
    ) -> list[spec.Message]:
        options = (
            LLMOptions(**options)
            if isinstance(options, dict)
            else (options or LLMOptions())
        )
        assert (
            options.response_format is None
        ), "response_format is not supported in generate_object"
        options.response_format = spec.ResponseFormatJson.from_model(return_type)
        r = self.generate(prompt, options=options)
        await r
        return r.new_messages

    def generate(
        self,
        prompt: str | spec.Message | Sequence[spec.Message],
        options: LLMOptionsUnion | None = None,
    ) -> ChatCompletion:
        options = (
            LLMOptions(**options)
            if isinstance(options, dict)
            else (options or LLMOptions())
        )

        async def gen() -> (
            AsyncGenerator[spec.AssistantMessage | spec.ToolMessage, None]
        ):
            tools = await self.__init_tools(options)
            messages = self.__prepare_messages(prompt)
            async with httpx.AsyncClient() as client:
                while True:
                    result = await self._provider.do_generate(
                        prompt=messages, tool_set=tools, options=options, client=client
                    )
                    c.usage += result.usage
                    c.new_messages.append(result.message)
                    yield result.message
                    # Add new messages
                    messages.append(result.message)
                    tool_calls = result.tool_calls
                    if result.finish_reason != "tool-calls":
                        break
                    # Call tools and continue
                    tool_msg, _, extra_msg = await self.__process_tool_calls(
                        tool_calls,
                        tools=tools,
                        parallel=options.parallel_tool_calls or False,
                    )
                    yield tool_msg
                    messages.append(tool_msg)
                    c.add_new_message(messages[-1])
                    if extra_msg:
                        messages.append(extra_msg)
                        c.add_new_message(messages[-1])

        async def gen_with_mcp_context() -> (
            AsyncGenerator[spec.AssistantMessage | spec.ToolMessage, None]
        ):
            from agentia.tools.mcp import MCPContext

            async with MCPContext() as _ctx:
                self._mcp_context = _ctx
                async for msg in gen():
                    yield msg
                self._mcp_context = None
            self._active_tools = None

        c = ChatCompletion(gen_with_mcp_context())
        return c

    @overload
    def stream(
        self,
        prompt: str | spec.Message | Sequence[spec.Message],
        /,
        events: Literal[False] = False,
        options: LLMOptionsUnion | None = None,
    ) -> ChatCompletionStream: ...

    @overload
    def stream(
        self,
        prompt: str | spec.Message | Sequence[spec.Message],
        /,
        events: Literal[True],
        options: LLMOptionsUnion | None = None,
    ) -> ChatCompletionEvents: ...

    def stream(
        self,
        prompt: str | spec.Message | Sequence[spec.Message],
        /,
        events: bool = False,
        options: LLMOptionsUnion | None = None,
    ) -> ChatCompletionStream | ChatCompletionEvents:
        options = (
            LLMOptions(**options)
            if isinstance(options, dict)
            else (options or LLMOptions())
        )

        async def gen() -> AsyncGenerator[StreamPart, None]:
            tools = await self.__init_tools(options)
            messages = self.__prepare_messages(prompt)
            last_finish_reason: spec.FinishReason = "unknown"

            async with httpx.AsyncClient() as client:
                while True:
                    tool_calls: list[spec.ToolCall] = []
                    parts: list[spec.AssistantMessagePart] = []
                    last_msg = ""
                    last_reasoning = ""
                    async for part in self._provider.do_stream(
                        messages, tools, options, client
                    ):

                        match part.type:
                            case "stream-start":
                                ...
                            case "text-start":
                                last_msg = ""
                            case "text-delta":
                                last_msg += part.delta
                            case "text-end":
                                parts.append(spec.MessagePartText(text=last_msg))
                            case "reasoning-start":
                                last_reasoning = ""
                            case "reasoning-delta":
                                last_reasoning += part.delta
                            case "reasoning-end":
                                parts.append(
                                    spec.MessagePartReasoning(text=last_reasoning)
                                )
                            case "tool-call":
                                parts.append(
                                    spec.MessagePartToolCall(
                                        tool_call_id=part.tool_call_id,
                                        tool_name=part.tool_name,
                                        input=part.input,
                                        provider_executed=part.provider_executed,
                                    )
                                )
                                if not part.provider_executed:
                                    tool_calls.append(part)

                        if part.type == "finish":
                            s.usage += part.usage
                            last_finish_reason = part.finish_reason
                        else:
                            yield part
                    messages.append(spec.AssistantMessage(content=parts))
                    s.new_messages.append(messages[-1])
                    if not tool_calls:
                        break
                    # Call tools and continue
                    tool_msg, tool_results, extra_msg = await self.__process_tool_calls(
                        tool_calls,
                        tools=tools,
                        parallel=options.parallel_tool_calls or False,
                    )
                    for tr in tool_results:
                        yield tr
                    messages.append(tool_msg)
                    s.add_new_message(messages[-1])
                    if extra_msg:
                        messages.append(extra_msg)
                        s.add_new_message(messages[-1])
            s.finish_reason = last_finish_reason
            yield StreamPartStreamEnd(usage=s.usage, finish_reason=last_finish_reason)

        async def gen_with_mcp_context() -> AsyncGenerator[StreamPart, None]:
            from agentia.tools.mcp import MCPContext

            async with MCPContext() as _ctx:
                self._mcp_context = _ctx
                async for part in gen():
                    yield part
                self._mcp_context = None

            self._active_tools = None

        if events:
            s = ChatCompletionEvents(gen_with_mcp_context())
        else:
            s = ChatCompletionStream(gen_with_mcp_context())
        return s
