import os
import re
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Literal,
    Optional,
    Sequence,
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


class GenerationOptions(BaseModel, arbitrary_types_allowed=True):
    max_output_tokens: int | None = None
    temperature: float | None = None
    stop_sequences: Sequence[str] | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    response_format: spec.ResponseFormat | None = None
    seed: int | None = None
    tool_choice: spec.ToolChoice | None = None
    provider_options: spec.ProviderOptions | None = None


class UnsupportedFunctionalityError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def get_provider(selector: str) -> "Provider":
    # if has no scheme, assume gateway
    if re.match(r"^\w+:", selector) is None:
        provider = "gateway"
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
            from .providers.openai import OpenAI

            if "AI_GATEWAY_API_KEY" not in os.environ:
                raise ValueError("AI_GATEWAY_API_KEY environment variable not set")

            return OpenAI(
                model=model,
                api_key=os.environ.get("AI_GATEWAY_API_KEY"),
                base_url="https://ai-gateway.vercel.sh/v1",
            )

        case "openrouter":
            from .providers.openai import OpenAI

            if "OPENROUTER_API_KEY" not in os.environ:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")

            return OpenAI(
                model=model,
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )

        case _:
            raise ValueError(f"Unknown provider: {provider}")


class LLM:
    def __init__(self, model: str, tools: Sequence[Tool] | None = None) -> None:
        self._provider = get_provider(model)
        self._provider.llm = self
        self._agent: Optional["Agent"] = None
        self._tools = ToolSet(tools or [])

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

    async def __init_tools(self, options: GenerationOptions) -> ToolSet:
        await self._tools.init()
        return self._tools

    async def generate_object[T: BaseModel | str | int | float | bool | None](
        self,
        prompt: str | spec.Message | Sequence[spec.Message],
        return_type: type[T],
        options: GenerationOptions | None = None,
    ) -> T:
        class Result[X](BaseModel):
            result: X = Field(..., description="The result", title="Result")

        class _Result(Result[return_type]): ...

        if inspect.isclass(return_type) and issubclass(return_type, BaseModel):
            response_format = return_type
        else:
            response_format = _Result

        json_string = ""
        options = options or GenerationOptions()
        options.response_format = spec.ResponseFormatJson(
            json_schema=response_format.model_json_schema(),
            name=return_type.__name__,
            description=f"JSON object matching the schema of {return_type.__name__}",
        )
        msg = await self.generate(prompt, options)
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
        options: GenerationOptions | None = None,
    ) -> ChatCompletion:
        options = options or GenerationOptions()

        async def gen():
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
        options = options or GenerationOptions()

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
