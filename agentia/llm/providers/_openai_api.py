import base64
from collections.abc import AsyncGenerator
from datetime import datetime
import json
import os
from typing import Any, Literal, Sequence, override
from uuid import uuid4
import httpx
from pydantic import HttpUrl

from agentia.tools.tools import ToolSet
from . import LLMOptions, GenerationResult, Provider

from ...spec import *
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
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

    def _to_oai_content_part(
        self, p: MessagePart, index: int
    ) -> ChatCompletionContentPartParam:
        match p.type:
            case "text":
                return ChatCompletionContentPartTextParam(type="text", text=p.text)
            case "file":
                if p.media_type.startswith("image/"):
                    if isinstance(p.data, str):
                        if p.data.startswith("data:"):
                            # this is a data URL
                            url = p.data
                        elif p.data.startswith("http://") or p.data.startswith(
                            "https://"
                        ):
                            # this is a URL
                            url = p.data
                        else:
                            # this is a base64 encoded string
                            url = f"data:{p.media_type};base64,{p.data}"
                    elif isinstance(p.data, HttpUrl):
                        url = str(p.data)
                    else:
                        assert isinstance(p.data, bytes)
                        base64_data = base64.b64encode(p.data).decode(encoding="utf-8")
                        url = f"data:{p.media_type};base64,{base64_data}"
                    detail = (p.provider_options or {}).get("imageDetail", None)
                    if detail not in ["auto", "low", "high"]:
                        detail = "auto"
                    return ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url={"url": url, "detail": detail},  # type: ignore
                    )
                if p.media_type.startswith("video/"):
                    if isinstance(p.data, str):
                        if p.data.startswith("data:"):
                            # this is a data URL
                            url = p.data
                        elif p.data.startswith("http://") or p.data.startswith(
                            "https://"
                        ):
                            # this is a URL
                            url = p.data
                        else:
                            # this is a base64 encoded string
                            url = f"data:{p.media_type};base64,{p.data}"
                    elif isinstance(p.data, HttpUrl):
                        url = str(p.data)
                    else:
                        assert isinstance(p.data, bytes)
                        base64_data = base64.b64encode(p.data).decode(encoding="utf-8")
                        url = f"data:{p.media_type};base64,{base64_data}"
                    return {"type": "video_url", "video_url": {"url": url}}  # type: ignore
                elif p.media_type.startswith("audio/"):
                    if isinstance(p.data, HttpUrl) or (
                        isinstance(p.data, str)
                        and (
                            p.data.startswith("http://")
                            or p.data.startswith("https://")
                        )
                    ):
                        raise ValueError("audio file parts with URLs")
                    if isinstance(p.data, str):
                        data = p.data
                    else:
                        assert isinstance(p.data, bytes)
                        data = p.data.decode(encoding="utf-8")
                    format: Literal["wav", "mp3"]
                    if p.media_type == "audio/wav":
                        format = "wav"
                    elif p.media_type == "audio/mpeg" or p.media_type == "audio/mp3":
                        format = "mp3"
                    else:
                        raise ValueError(
                            f"audio file parts with media type {p.media_type}"
                        )
                    return ChatCompletionContentPartInputAudioParam(
                        type="input_audio", input_audio={"data": data, "format": format}
                    )
                else:
                    if isinstance(p.data, str) and p.data.startswith("file-"):
                        # this is a file ID
                        return {"type": "file", "file": {"file_id": p.data}}
                    else:
                        url = None
                        if isinstance(p.data, str):
                            if p.data.startswith("data:"):
                                url = p.data
                            else:
                                # this is a base64 encoded string
                                url = f"data:{p.media_type};base64,{p.data}"
                        elif isinstance(p.data, HttpUrl):
                            url = str(p.data)
                        else:
                            assert isinstance(p.data, bytes)
                            base64_data = base64.b64encode(p.data).decode(
                                encoding="utf-8"
                            )
                            url = f"data:{p.media_type};base64,{base64_data}"
                        return {
                            "type": "file",
                            "file": {
                                "filename": p.filename or f"file-{index}",
                                "file_data": url,
                            },
                        }
            case _:
                raise ValueError(f"content part of type {p.type}")

    def _to_oai_messages(
        self, messages: list[Message]
    ) -> list[ChatCompletionMessageParam]:
        r: list[ChatCompletionMessageParam] = []
        for m in messages:
            if m.role == "system":
                r.append(
                    ChatCompletionSystemMessageParam(role="system", content=m.content)
                )
            elif m.role == "user":
                content = m.content_list
                if len(content) == 1 and content[0].type == "text":
                    r.append(
                        ChatCompletionUserMessageParam(
                            role="user", content=content[0].text
                        )
                    )
                    continue
                parts: list[ChatCompletionContentPartParam] = []
                for i, p in enumerate(content):
                    parts.append(self._to_oai_content_part(p, i))
                r.append(ChatCompletionUserMessageParam(role="user", content=parts))
            elif m.role == "assistant":
                text = ""
                reasoning = ""
                images: list[Any] = []
                tool_calls: list[ChatCompletionMessageToolCallParam] = []
                for i, p in enumerate(m.content_list):
                    if p.type == "text":
                        text += p.text
                    elif p.type == "reasoning":
                        reasoning += p.text
                    elif p.type == "file":
                        if p.media_type.startswith("image/"):
                            images.append(
                                {"type": "image_url", "image_url": {"url": p.to_url()}}
                            )
                        else:
                            raise ValueError(
                                f"Only image files are supported in assistant messages, got media type {p.media_type}"
                            )
                    elif p.type == "tool-call":
                        tool_calls.append(
                            {
                                "id": p.tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": p.tool_name,
                                    "arguments": json.dumps(p.input),
                                },
                            }
                        )
                p = ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=text if text else None,
                    tool_calls=tool_calls,
                )
                if reasoning:
                    p["reasoning"] = reasoning  # type: ignore
                if images:
                    p["images"] = images  # type: ignore
                r.append(p)
            elif m.role == "tool":
                for p in m.content:
                    val = p.serialize_output()
                    r.append(
                        ChatCompletionToolMessageParam(
                            role="tool", tool_call_id=p.tool_call_id, content=val
                        )
                    )
            else:
                raise ValueError(f"Unsupported message role {m.role}")
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
        tools: Sequence[Tool] | None,
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
                yield StreamPartStreamStart(
                    id=_gen_id(),
                    timestamp=datetime.fromtimestamp(chunk.created),
                    model_id=self.model,
                )
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
