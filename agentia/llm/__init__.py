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
from pydantic import AnyUrl

from agentia import spec
from agentia.llm.completion import ChatCompletion
from agentia.llm.stream import ChatCompletionEvents, ChatCompletionStream
from agentia.spec.prompt import ObjectType
from agentia.tools.tools import ToolSet, Tool

if TYPE_CHECKING:
    from agentia.agent import Agent
    from agentia.llm.providers import Provider


class GenerationOptions(TypedDict, total=False):
    max_output_tokens: int | None
    temperature: float | None
    stop_sequences: Sequence[str] | None
    top_p: float | None
    top_k: int | None
    presence_penalty: float | None
    frequency_penalty: float | None
    response_format: spec.ResponseFormat | None
    seed: int | None
    tool_choice: spec.ToolChoice | None
    provider_options: spec.ProviderOptions | None
    tools: Sequence[Tool] | ToolSet | None


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
        case _:
            raise ValueError(f"Unknown provider: {provider}")


class LLM:
    def __init__(self, model: str) -> None:
        self._provider = get_provider(model)
        self._provider.llm = self
        self._agent: Optional["Agent"] = None

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
        self, tool_calls: list[spec.ToolCall], tools: ToolSet | None = None
    ) -> tuple[spec.ToolMessage, list[spec.ToolResult], spec.Message | None]:
        assert tools is not None, "No tools provided"
        tm, tr, fr = await tools.run(self, tool_calls)
        if fr:
            msg = spec.UserMessage(
                content=[
                    spec.MessagePartText(text="[[TOOL_OUTPUT_FILES]]"),
                    *(
                        spec.MessagePartFile(
                            filename=f.id,
                            media_type=f.media_type,
                            data=f.data,
                        )
                        for f in fr
                    ),
                ]
            )
        else:
            msg = None
        return tm, tr, msg

    async def __init_tools(self, options: GenerationOptions | None) -> ToolSet:
        if options is None or "tools" not in options or options["tools"] is None:
            tools = ToolSet([])
        elif isinstance(options["tools"], ToolSet):
            tools = options["tools"]
        else:
            tools = ToolSet(options["tools"])
        await tools.init(self, self._agent)
        return tools

    async def generate_object[T: ObjectType](
        self,
        prompt: str | spec.Message | Sequence[spec.Message],
        type: type[T],
        options: GenerationOptions | None = None,
    ) -> T:
        msg = await self._generate_object_impl(prompt, type, options)
        return msg.parse(type)

    async def _generate_object_impl[T: ObjectType](
        self,
        prompt: str | spec.Message | Sequence[spec.Message],
        return_type: type[T],
        options: GenerationOptions | None = None,
    ) -> spec.AssistantMessage:
        options = options or {}
        options["response_format"] = spec.ResponseFormatJson.from_model(return_type)
        msg = await self.generate(prompt, options=options)
        return msg

    def generate(
        self,
        prompt: str | spec.Message | Sequence[spec.Message],
        options: GenerationOptions | None = None,
    ) -> ChatCompletion:
        options = options or {}

        async def gen() -> (
            AsyncGenerator[spec.AssistantMessage | spec.ToolMessage, None]
        ):
            tools = await self.__init_tools(options)
            messages = self.__prepare_messages(prompt)
            while True:
                result = await self._provider.do_generate(
                    prompt=messages, tool_set=tools, options=options
                )
                c.warnings.extend(result.warnings)
                c.usage += result.usage
                c.messages.append(result.message)
                yield result.message
                # Add new messages
                messages.append(result.message)
                tool_calls = result.tool_calls
                if result.finish_reason != "tool-calls":
                    break
                # Call tools and continue
                tool_msg, _, extra_msg = await self.__process_tool_calls(
                    tool_calls, tools=tools
                )
                yield tool_msg
                messages.append(tool_msg)
                c.messages.append(messages[-1])
                if extra_msg:
                    messages.append(extra_msg)
                    c.messages.append(messages[-1])

        c = ChatCompletion(gen())
        return c

    @overload
    def stream(
        self,
        prompt: str | spec.Message | Sequence[spec.Message],
        /,
        events: Literal[False] = False,
        options: GenerationOptions | None = None,
    ) -> ChatCompletionStream: ...

    @overload
    def stream(
        self,
        prompt: str | spec.Message | Sequence[spec.Message],
        /,
        events: Literal[True],
        options: GenerationOptions | None = None,
    ) -> ChatCompletionEvents: ...

    def stream(
        self,
        prompt: str | spec.Message | Sequence[spec.Message],
        /,
        events: bool = False,
        options: GenerationOptions | None = None,
    ) -> ChatCompletionStream | ChatCompletionEvents:
        options = options or {}

        async def gen() -> AsyncGenerator[spec.StreamPart, None]:
            tools = await self.__init_tools(options)
            messages = self.__prepare_messages(prompt)
            last_finish_reason: spec.FinishReason = "unknown"

            while True:
                tool_calls: list[spec.ToolCall] = []
                parts: list[spec.MessagePart] = []
                last_msg = ""
                last_reasoning = ""
                async for part in self._provider.do_stream(messages, tools, options):

                    match part.type:
                        case "stream-start":
                            s.warnings.extend(part.warnings)
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
                            parts.append(spec.MessagePartReasoning(text=last_reasoning))
                        case "file":
                            parts.append(
                                spec.MessagePartFile(
                                    data=part.data, media_type=part.media_type
                                )
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
                s.messages.append(messages[-1])
                if not tool_calls:
                    break
                # Call tools and continue
                tm, trs, extra_msg = await self.__process_tool_calls(
                    tool_calls, tools=tools
                )
                for tr in trs:
                    yield tr
                messages.append(tm)
                s.messages.append(messages[-1])
                if extra_msg:
                    messages.append(extra_msg)
                    s.messages.append(messages[-1])
            s.finish_reason = last_finish_reason
            yield spec.StreamPartFinish(usage=s.usage, finish_reason=last_finish_reason)

        if events:
            s = ChatCompletionEvents(gen())
        else:
            s = ChatCompletionStream(gen())
        return s
