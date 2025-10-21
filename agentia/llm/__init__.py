import os
import re
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Literal,
    Optional,
    Sequence,
    TypedDict,
    overload,
)
from pydantic import AnyUrl, BaseModel, Field
import inspect

from agentia import spec
from agentia.llm.completion import ChatCompletion
from agentia.llm.stream import ChatCompletionEvents, ChatCompletionStream
from agentia.llm.tools import ToolSet, Tool

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
            raise NotImplementedError("Use gateway instead")

        case "gateway":
            from .providers.gateway import Gateway

            return Gateway(model=model)
        case "openrouter":
            from .providers.openrouter import OpenRouter

            return OpenRouter(model=model)
        case _:
            raise ValueError(f"Unknown provider: {provider}")


class LLM:
    def __init__(self, model: str, options: GenerationOptions | None = None) -> None:
        self._provider = get_provider(model)
        self._provider.llm = self
        self._agent: Optional["Agent"] = None
        self.options = options or GenerationOptions()
        if tools := self.options.get("tools", None):
            if isinstance(tools, ToolSet):
                self._tools = tools
            else:
                self._tools = ToolSet(tools)
        else:
            self._tools = ToolSet([])

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
    ) -> tuple[spec.ToolMessage, list[spec.ToolResult]]:
        assert tools is not None, "No tools provided"
        tm, trs = await tools.run(self, tool_calls)
        return tm, trs

    async def __init_tools(self) -> ToolSet:
        await self._tools.init()
        return self._tools

    async def generate_object[T: BaseModel | str | int | float | bool | None](
        self, prompt: str | spec.Message | Sequence[spec.Message], return_type: type[T]
    ) -> T:
        class Result[X](BaseModel):
            result: X = Field(..., description="The result", title="Result")

        class _Result(Result[return_type]): ...

        if inspect.isclass(return_type) and issubclass(return_type, BaseModel):
            response_format = return_type
        else:
            response_format = _Result

        json_string = ""
        old_format = self.options.get("response_format", None)
        self.options["response_format"] = spec.ResponseFormatJson(
            json_schema=response_format.model_json_schema(),
            name=return_type.__name__,
            description=f"JSON object matching the schema of {return_type.__name__}",
        )
        msg = await self.generate(prompt)
        self.options["response_format"] = old_format
        for part in msg.content:
            if isinstance(part, spec.MessagePartText):
                json_string += part.text
        result = response_format.model_validate_json(json_string)
        if isinstance(result, _Result):
            return result.result
        else:
            return result

    def generate(
        self,
        prompt: str | spec.Message | Sequence[spec.Message],
    ) -> ChatCompletion:
        async def gen():
            tools = await self.__init_tools()
            messages = self.__prepare_messages(prompt)
            while True:
                result = await self._provider.do_generate(
                    prompt=messages, tool_set=tools, options=self.options
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
                messages.append(
                    (await self.__process_tool_calls(tool_calls, tools=tools))[0]
                )
                c.messages.append(messages[-1])

        c = ChatCompletion(gen())
        return c

    @overload
    def stream(
        self,
        prompt: str | spec.Message | Sequence[spec.Message],
        /,
        events: Literal[False] = False,
    ) -> ChatCompletionStream: ...

    @overload
    def stream(
        self,
        prompt: str | spec.Message | Sequence[spec.Message],
        /,
        events: Literal[True],
    ) -> ChatCompletionEvents: ...

    def stream(
        self,
        prompt: str | spec.Message | Sequence[spec.Message],
        /,
        events: bool = False,
    ) -> ChatCompletionStream | ChatCompletionEvents:
        async def gen() -> AsyncGenerator[spec.StreamPart, None]:
            tools = await self.__init_tools()
            messages = self.__prepare_messages(prompt)
            last_finish_reason: spec.FinishReason = "unknown"

            while True:
                tool_calls: list[spec.ToolCall] = []
                parts: list[spec.MessagePart] = []
                last_msg = ""
                last_reasoning = ""
                async for part in self._provider.do_stream(
                    messages, tools, self.options
                ):

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
                tm, trs = await self.__process_tool_calls(tool_calls, tools=tools)
                for tr in trs:
                    yield tr
                messages.append(tm)
                s.messages.append(messages[-1])
            s.finish_reason = last_finish_reason
            yield spec.StreamPartFinish(usage=s.usage, finish_reason=last_finish_reason)

        if events:
            s = ChatCompletionEvents(gen())
        else:
            s = ChatCompletionStream(gen())
        return s
