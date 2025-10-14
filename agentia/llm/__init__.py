import abc
from dataclasses import dataclass
import inspect
import json
import os
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Literal,
    Optional,
    Sequence,
    TypedDict,
    overload,
)
from openai.lib._parsing._completions import type_to_response_format_param  # type: ignore
from openai.lib._pydantic import _ensure_strict_json_schema
from pydantic import AnyUrl, BaseModel, Field, JsonValue, TypeAdapter


from ..tools import ToolRegistry
from ..message import AssistantMessage, Message, Event, is_event, is_message
from ..run import Run, MessageStream
from ..history import History
import re
import agentia.spec as spec

from dataclasses import dataclass

if TYPE_CHECKING:
    from agentia.llm.providers import Provider


@dataclass
class ModelOptions:
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    temperature: float | None = None
    reasoning_tokens: bool = True

    @staticmethod
    def from_dict(d: "ModelOptionsDict") -> "ModelOptions":
        reasoning_tokens = d.get("reasoning_tokens", True)
        return ModelOptions(
            frequency_penalty=d.get("frequency_penalty"),
            presence_penalty=d.get("presence_penalty"),
            temperature=d.get("temperature"),
            reasoning_tokens=(
                reasoning_tokens if isinstance(reasoning_tokens, bool) else True
            ),
        )


class ModelOptionsDict(TypedDict, total=False):
    frequency_penalty: float | None
    presence_penalty: float | None
    temperature: float | None
    reasoning_tokens: bool | None


class LLMBackend:
    def support_tools(self) -> bool:
        return True

    def __init__(
        self,
        *,
        name: str,
        model: str,
        tools: ToolRegistry,
        options: ModelOptions,
        history: History,
    ):
        self.name = name
        self.options = options or ModelOptions()
        self.model = model
        self.tools = tools
        self.history = history
        self.log = tools._agent.log

    @abc.abstractmethod
    def get_api_key(self) -> str: ...

    @classmethod
    def get_default_model(cls) -> str:
        raise NotImplementedError()

    @overload
    def run(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False] = False,
        events: Literal[False] = False,
        response_format: Any | None,
    ) -> Run[AssistantMessage]: ...

    @overload
    def run(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        events: Literal[False] = False,
        response_format: Any | None,
    ) -> Run[MessageStream]: ...

    @overload
    def run(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False] = False,
        events: Literal[True],
        response_format: Any | None,
    ) -> Run[AssistantMessage | Event]: ...

    @overload
    def run(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        events: Literal[True],
        response_format: Any | None,
    ) -> Run[MessageStream | Event]: ...

    def run(
        self,
        messages: Sequence[Message],
        *,
        stream: bool = False,
        events: bool = False,
        response_format: Any | None,
    ) -> (
        Run[MessageStream]
        | Run[AssistantMessage]
        | Run[MessageStream | Event]
        | Run[AssistantMessage | Event]
    ):
        a = self.tools._agent
        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            response_format = type_to_response_format_param(response_format)  # type: ignore
        if response_format is not None and not isinstance(response_format, dict):
            ta = TypeAdapter(response_format)
            schema = ta.json_schema()
            response_format = _ensure_strict_json_schema(schema, path=(), root=schema)
        if stream and events:
            return Run(
                a,
                self._run(
                    messages, stream=True, events=True, response_format=response_format
                ),  # type: ignore
            )
        elif stream:
            return Run(
                a, self._run(messages, stream=True, events=False, response_format=response_format)  # type: ignore
            )
        elif events:
            return Run(
                a, self._run(messages, stream=False, events=True, response_format=response_format)  # type: ignore
            )
        else:
            return Run(
                self.tools._agent, self._run(messages, stream=False, events=False, response_format=response_format)  # type: ignore
            )

    @overload
    async def _chat_completion_request(
        self,
        messages: list[Message],
        stream: Literal[False],
        response_format: Any | None,
    ) -> AssistantMessage: ...

    @overload
    async def _chat_completion_request(
        self,
        messages: Sequence[Message],
        stream: Literal[True],
        response_format: Any | None,
    ) -> MessageStream: ...

    async def _chat_completion_request(
        self,
        messages: Sequence[Message],
        stream: bool,
        response_format: Any | None,
    ) -> AssistantMessage | MessageStream:
        raise NotImplementedError()

    @overload
    async def _run(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False],
        events: Literal[False],
        response_format: Any | None,
    ) -> AsyncGenerator[AssistantMessage, None]: ...

    @overload
    async def _run(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        events: Literal[False],
        response_format: Any | None,
    ) -> AsyncGenerator[MessageStream, None]: ...

    @overload
    async def _run(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False],
        events: Literal[True],
        response_format: Any | None,
    ) -> AsyncGenerator[AssistantMessage | Event, None]: ...

    @overload
    async def _run(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        events: Literal[True],
        response_format: Any | None,
    ) -> AsyncGenerator[MessageStream | Event, None]: ...

    async def _run(
        self,
        messages: Sequence[Message],
        *,
        stream: bool,
        events: bool,
        response_format: Any | None,
    ) -> (
        AsyncGenerator[MessageStream, None]
        | AsyncGenerator[AssistantMessage, None]
        | AsyncGenerator[MessageStream | Event, None]
        | AsyncGenerator[AssistantMessage | Event, None]
    ):
        for m in messages:
            self.log.debug(m)
            self.history.add(m)
        # First completion request
        message: AssistantMessage
        trimmed_history = self.history.get_for_inference()
        if stream:
            s = await self._chat_completion_request(
                trimmed_history, stream=True, response_format=response_format
            )
            if await s._ensure_non_empty():
                yield s
            message = await s
        else:
            message = await self._chat_completion_request(
                trimmed_history, stream=False, response_format=response_format
            )
            if message.content or message.reasoning:
                yield message
        self.history.add(message)
        self.log.debug(message)
        # Run tools and submit results until convergence
        while len(message.tool_calls) > 0:
            # Run tools
            async for event in self.tools.call_tools(message.tool_calls):
                self.log.debug(event)
                if is_message(event):
                    self.history.add(event)
                else:
                    assert is_event(event), "Event must be a Event object"
                    if events:
                        self.history.add(event)
                        yield event
            trimmed_history = self.history.get_for_inference()
            # Submit results
            message: AssistantMessage
            if stream:
                r = await self._chat_completion_request(
                    trimmed_history, stream=True, response_format=response_format
                )
                if await r._ensure_non_empty():
                    yield r
                message = await r
            else:
                message = await self._chat_completion_request(
                    trimmed_history, stream=False, response_format=response_format
                )
                if message.content or message.reasoning:
                    yield message
            self.history.add(message)
            self.log.debug(message)


def get_default_provider() -> str:
    if p := os.environ.get("AGENTIA_LLM_BACKEND"):
        # The environment variable AGENTIA_LLM_BACKEND is set. use it.
        provider = p.strip().lower()
    elif "OPENROUTER_API_KEY" in os.environ:
        provider = "openrouter"
    elif "DEEPSEEK_API_KEY" in os.environ:
        provider = "deepseek"
    elif "GEMINI_API_KEY" in os.environ:
        provider = "google"
    else:
        provider = "openai"
    if provider not in [
        "openai",
        "openrouter",
        "deepseek",
        "google",
    ]:
        raise ValueError(f"Unknown provider: {provider}")
    return provider


def get_default_model() -> str:
    if model := os.environ.get("AGENTIA_MODEL"):
        return model
    match get_default_provider():
        case "openai":
            from .openai import OpenAIBackend

            return OpenAIBackend.get_default_model()
        case "openrouter":
            from .openrouter import OpenRouterBackend

            return OpenRouterBackend.get_default_model()
        case "deepseek":
            from .deepseek import DeepSeekBackend

            return DeepSeekBackend.get_default_model()
        case p:
            raise ValueError(f"Unknown provider: {p}")


def create_llm_backend(
    *,
    model: str,
    options: Optional["ModelOptions"],
    api_key: str | None,
    tools: ToolRegistry,
    history: History,
) -> LLMBackend:
    """
    :param model: The model name to use.
        * `openai/gpt-3.5-turbo` - by default this will use openrouter backend, if the environment variable `AGENTIA_LLM_BACKEND` is not set.
        * `[openai] gpt-3.5-turbo` - this will use openai backend
    """
    model = model.strip()
    if model.startswith("[") and "]" in model:
        # The model string has a [provider] prefix. parse it.
        provider = model.split("]", 1)[0][1:].strip().lower()
        model = model.split("]", 1)[1].strip()
    else:
        provider = get_default_provider()
    assert provider in [
        "openai",
        "openrouter",
        "deepseek",
        "google",
    ], f"Unknown provider: {provider}"
    if provider == "openai":
        from .openai import OpenAIBackend

        return OpenAIBackend(
            model=model,
            tools=tools,
            options=options or ModelOptions(),
            history=history,
            api_key=api_key,
        )
    elif provider == "deepseek":
        from .deepseek import DeepSeekBackend

        return DeepSeekBackend(
            model=model,
            tools=tools,
            options=options or ModelOptions(),
            history=history,
            api_key=api_key,
        )
    elif provider == "google":
        from .google import GoogleBackend

        return GoogleBackend(
            model=model,
            tools=tools,
            options=options or ModelOptions(),
            history=history,
            api_key=api_key,
        )
    else:
        from .openrouter import OpenRouterBackend

        return OpenRouterBackend(
            model=model,
            tools=tools,
            options=options or ModelOptions(),
            history=history,
            api_key=api_key,
        )


class LLMTool:
    def __init__(self, tool: spec.Tool, func: Callable[..., Any] | None) -> None:
        self.tool = tool
        self.func = func


type ToolSet = dict[str, LLMTool]


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
    tools: ToolSet | None = None
    tool_choice: spec.ToolChoice | None = None
    provider_options: spec.ProviderOptions | None = None


@dataclass
class TextGenerationResult:
    messages: Sequence[spec.Message]
    finish_reason: spec.FinishReason
    usage: spec.Usage
    warnings: Sequence[spec.Warning]


class ChatCompletionStream:
    def __init__(self) -> None:
        self.result = TextGenerationResult(
            messages=[], finish_reason="unknown", usage=spec.Usage(), warnings=[]
        )

    def __aiter__(self) -> AsyncIterator[spec.StreamPart]:
        raise NotImplementedError()


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
        print(uri.scheme, uri.host, uri.path)
        provider = uri.scheme
        model = (uri.host or "") + (uri.path or "")
    model = model.strip("/")
    print(f"Using provider: {provider}, model: {model}")
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
    def __init__(self, selector: str) -> None:
        self._provider = get_provider(selector)

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
        self, tool_calls: list[spec.ToolCall], options: GenerationOptions
    ) -> spec.ToolMessage:
        tool_results: list[spec.MessagePartToolResult] = []
        for c in tool_calls:
            f = options.tools.get(c.tool_name) if options.tools else None
            if not f:
                raise ValueError(f"Tool {c.tool_name} not found")
            if not f.func:
                raise ValueError(f"Tool {c.tool_name} has no function")
            try:
                args = json.loads(c.input)
                output = f.func(**args)  # type: ignore
                if inspect.isawaitable(output):
                    output = await output
                tool_results.append(
                    spec.MessagePartToolResult(
                        tool_call_id=c.tool_call_id,
                        tool_name=c.tool_name,
                        output=spec.ToolResultOutputJson(value=output),
                    )
                )
            except Exception as e:
                output: JsonValue = {"error": str(e)}
                tool_results.append(
                    spec.MessagePartToolResult(
                        tool_call_id=c.tool_call_id,
                        tool_name=c.tool_name,
                        output=spec.ToolResultOutputErrorJson(value=output),
                    )
                )
        return spec.ToolMessage(content=tool_results)

    async def generate_text(
        self,
        prompt: str | spec.Message | Sequence[spec.Message],
        options: GenerationOptions | None = None,
    ) -> TextGenerationResult:
        options = options or GenerationOptions()
        messages = self.__prepare_messages(prompt)
        usage = spec.Usage()
        while True:
            tool_calls: list[spec.ToolCall] = []
            result = await self._provider.do_generate(prompt=messages, options=options)
            usage += result.usage
            # Add new messages
            parts = []
            for c in result.content:
                if c.type == "text":
                    parts.append(spec.MessagePartText(text=c.text))
                elif c.type == "reasoning":
                    parts.append(spec.MessagePartReasoning(text=c.text))
                elif c.type == "file":
                    parts.append(
                        spec.MessagePartFile(data=c.data, media_type=c.media_type)
                    )
                elif c.type == "tool-call":
                    parts.append(
                        spec.MessagePartToolCall(
                            tool_call_id=c.tool_call_id,
                            tool_name=c.tool_name,
                            input=c.input,
                            provider_executed=c.provider_executed,
                        )
                    )
                    if not c.provider_executed:
                        tool_calls.append(c)
                else:
                    raise UnsupportedFunctionalityError(
                        f"Unsupported content type: {c.type}"
                    )
            if len(parts) > 0:
                messages.append(spec.AssistantMessage(content=parts))
            if result.finish_reason != "tool-calls":
                break
            # Call tools and continue
            messages.append(
                await self.__process_tool_calls(tool_calls, options=options)
            )

        return TextGenerationResult(
            messages=messages,
            finish_reason=result.finish_reason,
            usage=result.usage,
            warnings=result.warnings,
        )

    async def stream_text(
        self,
        prompt: str | spec.Message | Sequence[spec.Message],
        options: GenerationOptions | None = None,
    ) -> AsyncGenerator[spec.StreamPart, None]:
        options = options or GenerationOptions()
        messages = self.__prepare_messages(prompt)
        usage = spec.Usage()
        last_finish_reason: spec.FinishReason = "unknown"

        while True:
            tool_calls: list[spec.ToolCall] = []
            parts: list[spec.MessagePart] = []
            last_msg = ""
            last_reasoning = ""
            async for part in self._provider.do_stream(messages, options):

                match part.type:
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
                    usage += part.usage
                    last_finish_reason = part.finish_reason
                else:
                    yield part
            messages.append(spec.AssistantMessage(content=parts))
            if not tool_calls:
                break
            # Call tools and continue
            messages.append(
                await self.__process_tool_calls(tool_calls, options=options)
            )
        yield spec.StreamPartFinish(usage=usage, finish_reason=last_finish_reason)
