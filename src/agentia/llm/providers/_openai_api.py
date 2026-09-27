import os
from collections.abc import AsyncGenerator, Sequence
from typing import Any, override

import httpx
import openai
from openai.types.responses import Response, ResponseUsage

from agentia.tools.tools import ToolSet

from ...models import *
from . import GenerationResult, LLMOptions, Provider


class OpenAIAPIProvider(Provider):
    """Base provider for services that speak the OpenAI Responses API."""

    def __init__(
        self,
        name: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        if model.endswith(":think"):
            model = model[: -len(":think")]
            think = True
        else:
            think = False
        super().__init__(name=name, model=model)
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        if "OPENAI_BASE_URL" in os.environ:
            base_url = os.environ["OPENAI_BASE_URL"]
        self.api_key = api_key
        self.base_url = base_url
        self.extra_headers: dict[str, str] = {}
        self.extra_body: dict[str, Any] = {}
        self.__think_in_model_string = think

    def client(self, client: httpx.AsyncClient) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, http_client=client
        )

    def reasoning_enabled(self, options: LLMOptions) -> bool:
        if (r := options.reasoning) and r.enabled is not None:
            return r.enabled
        if self.__think_in_model_string:
            return True
        return os.environ.get("AGENTIA_REASONING_ENABLED", "").lower() in {
            "true",
            "1",
            "yes",
        }

    def get_reasoning_args(
        self,
        enabled: bool | None,
        effort: str | None,
        exclude: bool | None,
        max_tokens: int | None,
    ) -> dict[str, Any] | None:
        """Build the Responses API `reasoning` object.

        Only the fields the Responses API itself defines are emitted here. Gateways
        that accept more (e.g. OpenRouter's `enabled`/`exclude`/`max_tokens`) extend
        this in their own provider. `max_tokens` has no standard equivalent and is
        ignored unless a provider overrides this.
        """
        if not enabled:
            return None
        reasoning: dict[str, Any] = {}
        if effort is not None:
            reasoning["effort"] = effort
        if not exclude:
            # Ask for a summary so reasoning text is visible for models that only
            # expose reasoning through summaries (e.g. OpenAI's own models).
            reasoning["summary"] = "auto"
        return reasoning

    def get_reasoning_args_from_options(
        self, options: LLMOptions
    ) -> dict[str, Any] | None:
        enabled = self.reasoning_enabled(options)
        effort = os.environ.get("AGENTIA_REASONING_EFFORT")
        exclude = os.environ.get("AGENTIA_REASONING_EXCLUDE", "").lower() in {
            "true",
            "1",
            "yes",
        }
        if x := os.environ.get("AGENTIA_REASONING_MAX_TOKENS"):
            max_tokens = int(x)
        else:
            max_tokens = None
        if (r := options.reasoning) and enabled:
            if r.effort is not None:
                effort = r.effort
            if r.exclude is not None:
                exclude = r.exclude
            if r.max_tokens is not None:
                max_tokens = r.max_tokens
        return self.get_reasoning_args(enabled, effort, exclude, max_tokens)

    def _to_responses_input(self, items: Sequence[Item]) -> list[Any]:
        """Flatten the conversation into a Responses API `input` array."""
        return to_responses_input(items)

    def _get_finish_reason(self, response: Response | None) -> FinishReason:
        if response is None:
            return "unknown"
        if any(item.type == "function_call" for item in response.output):
            return "tool-calls"
        match response.status:
            case "completed":
                return "stop"
            case "incomplete":
                reason = (
                    response.incomplete_details.reason
                    if response.incomplete_details
                    else None
                )
                match reason:
                    case "max_output_tokens":
                        return "length"
                    case "content_filter":
                        return "content-filter"
                    case _:
                        return "other"
            case "failed":
                return "error"
            case "cancelled":
                return "interrupted"
            case _:
                return "unknown"

    def _prepare_tools(
        self,
        tools: Sequence[ToolSchema] | None,
        tool_choice: ToolChoice | None,
    ) -> tuple[list[Any] | None, Any | None]:
        if not tools:
            return None, None
        oai_tools: list[Any] = [t.to_responses_tool() for t in tools]
        if not tool_choice:
            return oai_tools, None
        match tool_choice:
            case "none" | "auto" | "required":
                return oai_tools, tool_choice
            case _:
                assert isinstance(tool_choice, str)
                return oai_tools, {"type": "function", "name": tool_choice}

    def _get_usage(self, u: ResponseUsage | None) -> Usage:
        if not u:
            return Usage()
        return Usage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            total_tokens=u.total_tokens,
            reasoning_tokens=(
                u.output_tokens_details.reasoning_tokens
                if u.output_tokens_details
                else None
            ),
            cached_input_tokens=(
                u.input_tokens_details.cached_tokens if u.input_tokens_details else None
            ),
        )

    def _prepare_args(
        self,
        instructions: str,
        items: Sequence[Item],
        tool_set: ToolSet,
        options: LLMOptions,
    ) -> dict[str, Any]:
        rf = options.response_format
        if rf and not isinstance(rf, (ResponseFormatJson, ResponseFormatText)):
            rf = ResponseFormatJson.from_model(rf)
        text = {"format": rf.to_responses_text_format()} if rf else None
        schema = tool_set.get_schema()
        tools, tool_choice = self._prepare_tools(schema, options.tool_choice)
        args: dict[str, Any] = {
            "model": self.model,
            "input": self._to_responses_input(items),
            "instructions": instructions.strip() or None,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "max_output_tokens": options.max_output_tokens,
            "text": text,
            "tools": tools,
            "tool_choice": tool_choice,
            # Only meaningful alongside tools, and rejected by some providers without them.
            "parallel_tool_calls": options.parallel_tool_calls if tools else None,
            # Never persist conversations server-side; the history lives in the agent.
            "store": False,
        }
        if self.reasoning_enabled(options):
            # Without server-side storage, reasoning can only be replayed if the
            # provider hands back the encrypted payload.
            args["include"] = ["reasoning.encrypted_content"]
        for k in list(args.keys()):
            if args[k] is None:
                del args[k]
        return args

    def _get_extra_body(self, options: LLMOptions) -> dict[str, Any]:
        body: dict[str, Any] = {}
        body.update(self.extra_body)
        if reasoning_args := self.get_reasoning_args_from_options(options):
            body.update({"reasoning": reasoning_args})
        # The Responses API has no first-class fields for these, so they are only
        # forwarded when explicitly set, for gateways that still accept them.
        for key, value in (
            ("frequency_penalty", options.frequency_penalty),
            ("presence_penalty", options.presence_penalty),
            ("seed", options.seed),
            ("stop", options.stop_sequences),
        ):
            if value is not None:
                body[key] = value
        if options.extra_body:
            body.update(options.extra_body)
        return body

    def _timeout(self) -> float | None:
        if t := os.environ.get("AGENTIA_TIMEOUT"):
            return float(t)
        return None

    def _parse_reasoning_item(self, item: Any) -> Reasoning:
        return Reasoning(
            id=item.id,
            content=[
                ReasoningText(text=c.text)
                for c in (getattr(item, "content", None) or [])
                if getattr(c, "type", None) == "reasoning_text"
            ],
            summary=[
                SummaryText(text=s.text)
                for s in (item.summary or [])
                if getattr(s, "type", None) == "summary_text"
            ],
            encrypted_content=getattr(item, "encrypted_content", None),
            signature=getattr(item, "signature", None),
            format=getattr(item, "format", None),
        )

    def _parse_message_item(self, item: Any) -> Message:
        content: list[OutputContent] = []
        for c in item.content:
            match c.type:
                case "output_text":
                    content.append(
                        OutputText(
                            text=c.text,
                            annotations=[
                                UrlCitation(
                                    title=a.title or "",
                                    url=a.url,
                                    start_index=a.start_index or 0,
                                    end_index=a.end_index or 0,
                                )
                                for a in (c.annotations or [])
                                if a.type == "url_citation"
                            ],
                        )
                    )
                case "refusal":
                    content.append(Refusal(refusal=c.refusal))
        return Message(
            role="assistant", id=item.id, content=content, status="completed"
        )

    def _parse_output(self, response: Response, reasoning_enabled: bool) -> list[Item]:
        """Convert a response's `output` array into conversation items."""
        items: list[Item] = []
        for item in response.output:
            match item.type:
                case "reasoning":
                    if not reasoning_enabled:
                        continue
                    r = self._parse_reasoning_item(item)
                    if r.display_text or r.encrypted_content or r.signature:
                        items.append(r)
                case "message":
                    items.append(self._parse_message_item(item))
                case "function_call":
                    items.append(
                        FunctionCall(
                            id=item.id,
                            call_id=item.call_id,
                            name=item.name,
                            arguments=item.arguments or "{}",
                            status="completed",
                        )
                    )
        return items

    def _response_info(self, response: Response | None) -> ResponseInfo:
        if response is None:
            return ResponseInfo()
        return ResponseInfo(
            id=response.id,
            status=response.status,
            usage=self._get_usage(response.usage),
            finish_reason=self._get_finish_reason(response),
        )

    @override
    async def generate(
        self,
        instructions: str,
        items: Sequence[Item],
        tools: ToolSet,
        options: LLMOptions,
        client: httpx.AsyncClient,
    ) -> GenerationResult:
        args = self._prepare_args(instructions, items, tools, options)
        response = await self.client(client).responses.create(
            **args,
            extra_headers=self.extra_headers,
            extra_body=self._get_extra_body(options),
            stream=False,
            timeout=self._timeout(),
        )
        return GenerationResult(
            items=self._parse_output(response, self.reasoning_enabled(options)),
            finish_reason=self._get_finish_reason(response),
            usage=self._get_usage(response.usage),
            response_id=response.id,
        )

    @override
    async def stream(
        self,
        instructions: str,
        items: Sequence[Item],
        tools: ToolSet,
        options: LLMOptions,
        client: httpx.AsyncClient,
    ) -> AsyncGenerator[StreamEvent, None]:
        args = self._prepare_args(instructions, items, tools, options)
        events = await self.client(client).responses.create(
            **args,
            extra_headers=self.extra_headers,
            extra_body=self._get_extra_body(options),
            stream=True,
            timeout=self._timeout(),
        )

        reasoning_enabled = self.reasoning_enabled(options)
        seq = 0
        response: Response | None = None

        def next_seq() -> int:
            nonlocal seq
            seq += 1
            return seq

        async for event in events:
            # Matching on `event.type` directly keeps the union narrowed.
            match event.type:
                case "response.created":
                    yield ResponseCreatedEvent(
                        sequence_number=next_seq(),
                        response=ResponseInfo(
                            id=event.response.id, status=event.response.status
                        ),
                    )

                case "response.in_progress":
                    yield ResponseInProgressEvent(
                        sequence_number=next_seq(),
                        response=ResponseInfo(
                            id=event.response.id, status=event.response.status
                        ),
                    )

                case "response.output_item.added":
                    if event.item.type == "reasoning" and not reasoning_enabled:
                        continue
                    if parsed := self._parse_stream_item(event.item):
                        yield ResponseOutputItemAddedEvent(
                            sequence_number=next_seq(),
                            output_index=event.output_index,
                            item=parsed,
                        )

                case "response.output_item.done":
                    if event.item.type == "reasoning" and not reasoning_enabled:
                        continue
                    if parsed := self._parse_stream_item(event.item):
                        yield ResponseOutputItemDoneEvent(
                            sequence_number=next_seq(),
                            output_index=event.output_index,
                            item=parsed,
                        )

                case "response.content_part.added" | "response.content_part.done":
                    part = self._parse_content_part(event.part)
                    if part is None:
                        continue
                    cls = (
                        ResponseContentPartAddedEvent
                        if event.type == "response.content_part.added"
                        else ResponseContentPartDoneEvent
                    )
                    yield cls(
                        sequence_number=next_seq(),
                        item_id=event.item_id,
                        output_index=event.output_index,
                        content_index=event.content_index,
                        part=part,
                    )

                case "response.output_text.delta":
                    yield ResponseOutputTextDeltaEvent(
                        sequence_number=next_seq(),
                        item_id=event.item_id,
                        output_index=event.output_index,
                        content_index=event.content_index,
                        delta=event.delta,
                    )

                case "response.output_text.done":
                    yield ResponseOutputTextDoneEvent(
                        sequence_number=next_seq(),
                        item_id=event.item_id,
                        output_index=event.output_index,
                        content_index=event.content_index,
                        text=event.text,
                    )

                case "response.output_text.annotation.added":
                    a = event.annotation
                    if isinstance(a, dict) and a.get("type") == "url_citation":
                        yield ResponseOutputTextAnnotationAddedEvent(
                            sequence_number=next_seq(),
                            item_id=event.item_id,
                            output_index=event.output_index,
                            content_index=event.content_index,
                            annotation_index=event.annotation_index,
                            annotation=UrlCitation(
                                title=str(a.get("title") or ""),
                                url=str(a.get("url") or ""),
                                start_index=int(a.get("start_index") or 0),
                                end_index=int(a.get("end_index") or 0),
                            ),
                        )

                case "response.refusal.delta":
                    yield ResponseRefusalDeltaEvent(
                        sequence_number=next_seq(),
                        item_id=event.item_id,
                        output_index=event.output_index,
                        content_index=event.content_index,
                        delta=event.delta,
                    )

                case "response.refusal.done":
                    yield ResponseRefusalDoneEvent(
                        sequence_number=next_seq(),
                        item_id=event.item_id,
                        output_index=event.output_index,
                        content_index=event.content_index,
                        refusal=event.refusal,
                    )

                case "response.reasoning_text.delta":
                    if not reasoning_enabled:
                        continue
                    yield ResponseReasoningTextDeltaEvent(
                        sequence_number=next_seq(),
                        item_id=event.item_id,
                        output_index=event.output_index,
                        content_index=event.content_index,
                        delta=event.delta,
                    )

                case "response.reasoning_text.done":
                    if not reasoning_enabled:
                        continue
                    yield ResponseReasoningTextDoneEvent(
                        sequence_number=next_seq(),
                        item_id=event.item_id,
                        output_index=event.output_index,
                        content_index=event.content_index,
                        text=event.text,
                    )

                case "response.reasoning_summary_text.delta":
                    if not reasoning_enabled:
                        continue
                    yield ResponseReasoningSummaryTextDeltaEvent(
                        sequence_number=next_seq(),
                        item_id=event.item_id,
                        output_index=event.output_index,
                        summary_index=event.summary_index,
                        delta=event.delta,
                    )

                case "response.reasoning_summary_text.done":
                    if not reasoning_enabled:
                        continue
                    yield ResponseReasoningSummaryTextDoneEvent(
                        sequence_number=next_seq(),
                        item_id=event.item_id,
                        output_index=event.output_index,
                        summary_index=event.summary_index,
                        text=event.text,
                    )

                case (
                    "response.reasoning_summary_part.added"
                    | "response.reasoning_summary_part.done"
                ):
                    if not reasoning_enabled:
                        continue
                    cls2 = (
                        ResponseReasoningSummaryPartAddedEvent
                        if event.type == "response.reasoning_summary_part.added"
                        else ResponseReasoningSummaryPartDoneEvent
                    )
                    yield cls2(
                        sequence_number=next_seq(),
                        item_id=event.item_id,
                        output_index=event.output_index,
                        summary_index=event.summary_index,
                        part=SummaryText(text=event.part.text),
                    )

                case "response.function_call_arguments.delta":
                    yield ResponseFunctionCallArgumentsDeltaEvent(
                        sequence_number=next_seq(),
                        item_id=event.item_id,
                        output_index=event.output_index,
                        delta=event.delta,
                    )

                case "response.function_call_arguments.done":
                    yield ResponseFunctionCallArgumentsDoneEvent(
                        sequence_number=next_seq(),
                        item_id=event.item_id,
                        output_index=event.output_index,
                        arguments=event.arguments,
                    )

                case "response.completed" | "response.incomplete" | "response.failed":
                    response = event.response
                    info = self._response_info(response)
                    match event.type:
                        case "response.completed":
                            yield ResponseCompletedEvent(
                                sequence_number=next_seq(), response=info
                            )
                        case "response.incomplete":
                            yield ResponseIncompleteEvent(
                                sequence_number=next_seq(), response=info
                            )
                        case _:
                            yield ResponseFailedEvent(
                                sequence_number=next_seq(), response=info
                            )

                case "error":
                    yield ErrorEvent(
                        sequence_number=next_seq(),
                        message=event.message,
                        code=event.code,
                        param=event.param,
                    )

        if response is None:
            # The provider ended the stream without a terminal response event.
            yield ResponseCompletedEvent(
                sequence_number=next_seq(), response=ResponseInfo()
            )

    def _parse_stream_item(self, item: Any) -> Item | None:
        """Convert one streamed output item into a conversation item."""
        match item.type:
            case "reasoning":
                return self._parse_reasoning_item(item)
            case "message":
                return self._parse_message_item(item)
            case "function_call":
                return FunctionCall(
                    id=item.id,
                    call_id=item.call_id,
                    name=item.name,
                    arguments=item.arguments or "{}",
                    status="completed",
                )
            case _:
                return None

    def _parse_content_part(self, part: Any) -> Content | None:
        match getattr(part, "type", None):
            case "output_text":
                return OutputText(text=part.text)
            case "refusal":
                return Refusal(refusal=part.refusal)
            case _:
                return None
