from collections.abc import AsyncGenerator
import json
import os
from typing import Any, Literal, Sequence, override
from uuid import uuid4
import httpx
from agentia.tools.tools import ToolSet
from . import LLMOptions, GenerationResult, Provider
from ...spec import *
from openai.types.chat import (
    ChatCompletionToolParam,
    ChatCompletionToolChoiceOptionParam,
)
import openai
from openai.types import CompletionUsage


def _gen_id() -> str:
    return uuid4().hex


class OpenAIAPIProvider(Provider):
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        if model.endswith(":think"):
            model = model[: -len(":think")]
            think = True
        else:
            think = False
        super().__init__(provider=provider, model=model)
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        if "OPENAI_BASE_URL" in os.environ:
            base_url = os.environ["OPENAI_BASE_URL"]
        self.api_key = api_key
        self.base_url = base_url
        self.extra_headers: dict[str, str] = {}
        self.extra_body: dict[str, Any] = {}
        if think:
            self.reasoning = True
            self.enable_reasoning()
        else:
            self.reasoning = False

    def client(self, client: httpx.AsyncClient) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, http_client=client
        )

    def enable_reasoning(self) -> None:
        self.extra_body["reasoning"] = {
            "enabled": True,
            "effort": "high",
        }

    def _to_oai_messages(self, messages: list[Message]) -> list[Any]:
        r: list[Any] = []
        for m in messages:
            oai_m = m.to_openai_format()
            if isinstance(oai_m, list):
                r.extend(oai_m)
            else:
                r.append(oai_m)
        return r

    def _get_finish_reason(
        self,
        choice: Literal[
            "stop", "length", "tool_calls", "content_filter", "function_call"
        ],
    ) -> FinishReason:
        match choice:
            case "stop":
                return "stop"
            case "length":
                return "length"
            case "tool_calls" | "function_call":
                return "tool-calls"
            case "content_filter":
                return "content-filter"
            case _:
                return "unknown"

    def _prepare_tools(
        self,
        tools: Sequence[ToolSchema] | None,
        tool_choice: ToolChoice | None,
    ) -> tuple[
        list[ChatCompletionToolParam] | None, ChatCompletionToolChoiceOptionParam | None
    ]:
        oai_tools: list[ChatCompletionToolParam] = []
        if not tools:
            return None, None
        for t in tools:
            match t.type:
                case "function":
                    f = {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,  # type: ignore
                    }
                    for k in list(f.keys()):
                        if f[k] is None:
                            del f[k]
                    oai_tools.append(
                        {
                            "type": "function",
                            "function": f,  # type: ignore
                        }
                    )
        if not tool_choice:
            return oai_tools, None
        match tool_choice:
            case "none":
                return oai_tools, "none"
            case "auto":
                return oai_tools, "auto"
            case "required":
                return oai_tools, "required"
            case _:
                assert isinstance(tool_choice, str)
                tc: ChatCompletionToolChoiceOptionParam = {
                    "type": "function",
                    "function": {"name": tool_choice},
                }
                return oai_tools, tc

    def _get_usage(self, u: CompletionUsage | None) -> Usage:
        if not u:
            return Usage()
        return Usage(
            input_tokens=u.prompt_tokens,
            output_tokens=u.completion_tokens,
            total_tokens=u.total_tokens,
            reasoning_tokens=(
                u.completion_tokens_details.reasoning_tokens
                if u.completion_tokens_details
                else None
            ),
            cached_input_tokens=(
                u.prompt_tokens_details.cached_tokens
                if u.prompt_tokens_details
                else None
            ),
        )

    def _prepare_args(
        self, messages: list[Message], tool_set: ToolSet, options: LLMOptions
    ) -> dict[str, Any]:
        msgs = self._to_oai_messages(messages)
        rf = options.response_format
        if not rf or rf.type == "text":
            response_format = None
        elif not rf.json_schema:
            response_format = {"type": "json_object"}
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "schema": rf.json_schema,
                    "strict": True,
                    "name": rf.name or "response",
                    "description": rf.description,
                },
            }
        schema = tool_set.get_schema()
        tools, tool_choice = self._prepare_tools(schema, options.tool_choice)
        args: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "max_tokens": options.max_output_tokens,
            "frequency_penalty": options.frequency_penalty,
            "presence_penalty": options.presence_penalty,
            "response_format": response_format,
            "stop": options.stop_sequences,
            "seed": options.seed,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        for k in list(args.keys()):
            if args[k] is None:
                del args[k]
        return args

    def _get_extra_body(self, options: LLMOptions) -> dict[str, Any]:
        body: dict[str, Any] = {}
        body.update(self.extra_body)
        if options.provider_options:
            body.update(options.provider_options)
        return body

    @override
    async def generate(
        self,
        messages: list[Message],
        tools: ToolSet,
        options: LLMOptions,
        client: httpx.AsyncClient,
    ) -> GenerationResult:
        args = self._prepare_args(messages, tools, options)
        if t := os.environ.get("AGENTIA_TIMEOUT"):
            timeout = float(t)
        else:
            timeout = None
        response = await self.client(client).chat.completions.create(
            **args,
            extra_headers=self.extra_headers,
            extra_body=self._get_extra_body(options),
            stream=False,
            timeout=timeout,
        )
        choice = response.choices[0]
        parts: Sequence[AssistantMessagePart] = []

        if self.reasoning and hasattr(choice.message, "reasoning"):
            reasoning_text: str = choice.message.reasoning  # type: ignore
            if reasoning_text and reasoning_text.strip() != "":
                parts.append(MessagePartReasoning(text=reasoning_text))

        text = choice.message.content
        if text:
            parts.append(MessagePartText(text=text))
        for tc in choice.message.tool_calls or []:
            assert tc.type == "function"
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            parts.append(
                MessagePartToolCall(
                    tool_call_id=tc.id or _gen_id(),
                    tool_name=tc.function.name,
                    input=args,
                )
            )

        annotations: list[Annotation] | None = None
        for a in choice.message.annotations or []:
            if not annotations:
                annotations = []
            annotations.append(
                Annotation(
                    title=a.url_citation.title,
                    url=a.url_citation.url,
                    start=a.url_citation.start_index,
                    end=a.url_citation.end_index,
                )
            )

        return GenerationResult(
            message=AssistantMessage(content=parts, annotations=annotations),
            finish_reason=self._get_finish_reason(choice.finish_reason),
            usage=self._get_usage(response.usage),
            provider_metadata=None,
        )

    @override
    async def stream(
        self,
        messages: list[Message],
        tools: ToolSet,
        options: LLMOptions,
        client: httpx.AsyncClient,
    ) -> AsyncGenerator[StreamPart, None]:
        args = self._prepare_args(messages, tools, options)
        if t := os.environ.get("AGENTIA_TIMEOUT"):
            timeout = float(t)
        else:
            timeout = None
        response = await self.client(client).chat.completions.create(
            **args,
            extra_headers=self.extra_headers,
            extra_body=self._get_extra_body(options),
            stream=True,
            stream_options={
                "include_usage": True,
            },
            timeout=timeout,
        )
        started = False
        streaming_text = False
        streaming_reasoning = False
        tool_calls: list[ToolCall | None] = []
        tool_call_partial_inputs: dict[str, str] = {}
        finished = False
        async for chunk in response:
            if finished:
                continue
            if not started:
                started = True
                yield StreamPartStreamStart(id=_gen_id(), model_id=self.model)
            if not chunk.choices:
                yield StreamPartStreamEnd(
                    usage=self._get_usage(chunk.usage), finish_reason="stop"
                )
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if rd := getattr(delta, "reasoning", None):
                if not self.reasoning:
                    continue
                if not streaming_reasoning:
                    streaming_reasoning = True
                    yield StreamPartReasoningStart(id=_gen_id())
                yield StreamPartReasoningDelta(id=_gen_id(), delta=rd)
            elif streaming_reasoning:
                streaming_reasoning = False
                yield StreamPartReasoningEnd(id=_gen_id())

            if delta.content:
                if not streaming_text:
                    streaming_text = True
                    yield StreamPartTextStart(id=_gen_id())
                yield StreamPartTextDelta(id=_gen_id(), delta=delta.content)
            elif streaming_text:
                streaming_text = False
                yield StreamPartTextEnd(id=_gen_id())

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    index = tc.index
                    if index >= len(tool_calls) or tool_calls[index] is None:
                        tool_calls.extend([None] * (index - len(tool_calls) + 1))
                        assert tc.id and tc.function and tc.function.name
                        tool_calls[index] = ToolCall(
                            tool_call_id=tc.id, tool_name=tc.function.name, input={}
                        )
                        tool_call_partial_inputs[tc.id] = ""
                    tc_obj = tool_calls[index]
                    assert tc_obj is not None
                    if tc.function and tc.function.arguments:
                        assert isinstance(
                            tool_call_partial_inputs[tc_obj.tool_call_id], str
                        )
                        tool_call_partial_inputs[tc_obj.tool_call_id] += tc.function.arguments or ""  # type: ignore

            if choice.finish_reason:
                if streaming_text:
                    streaming_text = False
                    yield StreamPartTextEnd(id=_gen_id())
                for i, tc in enumerate(tool_calls):
                    if not tc:
                        continue
                    assert isinstance(tool_call_partial_inputs[tc.tool_call_id], str)
                    tc.input = json.loads(
                        tool_call_partial_inputs[tc.tool_call_id] or "{}"
                    )
                    yield tc
                yield StreamPartStreamEnd(
                    usage=self._get_usage(chunk.usage),
                    finish_reason=self._get_finish_reason(choice.finish_reason),
                )
                finished = True
