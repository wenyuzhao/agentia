"""The OpenAI Responses API over plain httpx.

Always streams (`stream: true`) and never stores (`store: false`): the full context is
sent every time. Handles what real gateways do in practice: SSE comments and `[DONE]`,
proxies that answer with a plain JSON `Response`, mid-stream error frames (including
OpenRouter's Chat-Completions-shaped ones), streams that never send `output_item.done`
or a terminal event, and OpenRouter's `usage.cost`.

Failed calls are retried with jittered backoff (408/429/5xx and network errors) and then
handed to the fallback models, but never once output has streamed to the caller.
"""

import asyncio
import copy
import json
import os
import random
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, override

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from ...models import *
from . import LLMError, LLMOptions, ModelRequest, Provider

DEFAULT_MAX_RETRIES = 3
_BENIGN_CODES = ("invalid", "permission", "policy", "auth")
# Model catalogs (id → context length) by URL, with the time they were fetched.
_models_cache: dict[str, tuple[float, dict[str, int]]] = {}
_MODELS_TTL_S = 3600
_CONTENT = TypeAdapter(Content)


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in {"true", "1", "yes"}


def _error_from(message: Any, code: Any) -> LLMError:
    status = code if isinstance(code, int) else None
    if status is not None:
        retryable = status in (408, 429) or status >= 500
    else:
        retryable = not any(b in str(code or "").lower() for b in _BENIGN_CODES)
    suffix = f" ({code})" if code is not None else ""
    return LLMError(
        f"provider error: {message or 'unknown'}{suffix}", status, retryable
    )


def parse_usage(u: dict[str, Any] | None) -> Usage:
    if not u:
        return Usage()
    out_d = u.get("output_tokens_details") or {}
    in_d = u.get("input_tokens_details") or {}
    cost = u.get("cost")
    return Usage(
        input_tokens=u.get("input_tokens"),
        output_tokens=u.get("output_tokens"),
        total_tokens=u.get("total_tokens"),
        reasoning_tokens=out_d.get("reasoning_tokens"),
        cached_input_tokens=in_d.get("cached_tokens"),
        cost=float(cost) if isinstance(cost, (int, float)) else None,
    )


def parse_item(o: dict[str, Any]) -> Item | None:
    """One Responses API output item → a conversation item (None for other types)."""
    match o.get("type"):
        case "message":
            content: list[OutputContent] = []
            for c in o.get("content") or []:
                match c.get("type"):
                    case "output_text":
                        content.append(
                            OutputText(
                                text=c.get("text") or "",
                                annotations=[
                                    UrlCitation(
                                        title=str(a.get("title") or ""),
                                        url=str(a.get("url") or ""),
                                        start_index=int(a.get("start_index") or 0),
                                        end_index=int(a.get("end_index") or 0),
                                    )
                                    for a in c.get("annotations") or []
                                    if isinstance(a, dict)
                                    and a.get("type") == "url_citation"
                                ],
                            )
                        )
                    case "refusal":
                        content.append(Refusal(refusal=c.get("refusal") or ""))
            return Message(
                role="assistant",
                id=o.get("id"),
                content=content,
                status=o.get("status") or "completed",
            )
        case "function_call":
            args = o.get("arguments")
            if not isinstance(args, str):
                args = json.dumps(args) if args is not None else "{}"
            return FunctionCall(
                id=o.get("id"),
                call_id=o.get("call_id") or f"call_{uuid.uuid4().hex}",
                name=o.get("name") or "",
                arguments=args or "{}",
                status="completed",
            )
        case "reasoning":
            return Reasoning(
                id=o.get("id"),
                content=[
                    ReasoningText(text=c.get("text") or "")
                    for c in o.get("content") or []
                    if c.get("type") == "reasoning_text"
                ],
                summary=[
                    SummaryText(text=s.get("text") or "")
                    for s in o.get("summary") or []
                    if s.get("type") == "summary_text"
                ],
                encrypted_content=o.get("encrypted_content"),
                signature=o.get("signature"),
                format=o.get("format"),
            )
    return None


def finish_reason_of(
    status: str | None, incomplete: Any, items: Sequence[Item]
) -> FinishReason:
    if any(isinstance(i, FunctionCall) for i in items):
        return "tool-calls"
    match status:
        case "completed" | None:
            return "stop"
        case "incomplete":
            reason = incomplete.get("reason") if isinstance(incomplete, dict) else None
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


@dataclass
class _Partial:
    """An output item being assembled from deltas."""

    raw: dict[str, Any]
    text: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)


class StreamAccumulator:
    """Raw Responses API events → agentia stream events, plus the final output items."""

    def __init__(self, reasoning_enabled: bool) -> None:
        self.reasoning_enabled = reasoning_enabled
        self.response_id: str | None = None
        self.model: str | None = None
        self.partial: dict[int, _Partial] = {}
        self.done: dict[int, Item] = {}
        self.finished = False
        self.content_seen = False

    def _index(self, e: dict[str, Any]) -> int:
        i = e.get("output_index")
        if isinstance(i, int):
            return i
        item_id = e.get("item_id")
        for k, p in self.partial.items():
            if item_id and p.raw.get("id") == item_id:
                return k
        return max(self.partial, default=0)

    def _partial(self, e: dict[str, Any], kind: str) -> _Partial:
        i = self._index(e)
        if i not in self.partial:
            self.partial[i] = _Partial(raw={"type": kind, "id": e.get("item_id")})
        return self.partial[i]

    def _filtered(self, item: Item | None) -> Item | None:
        if isinstance(item, Reasoning) and not self.reasoning_enabled:
            return None
        return item

    def _keep(self, item: Item) -> bool:
        match item:
            case Reasoning():
                return self.reasoning_enabled and bool(
                    item.display_text or item.encrypted_content or item.signature
                )
            case FunctionCall():
                return bool(item.name)
        return True

    @staticmethod
    def _validate[E: BaseModel](event_cls: type[E], e: dict[str, Any]) -> list[E]:
        try:
            return [event_cls.model_validate(e)]
        except ValidationError:
            return []

    def _info(self, r: dict[str, Any], output: Sequence[Item] = ()) -> ResponseInfo:
        return ResponseInfo(
            id=r.get("id") or self.response_id,
            model=r.get("model") or self.model,
            status=r.get("status"),
            usage=parse_usage(r.get("usage")),
            finish_reason=finish_reason_of(
                r.get("status"), r.get("incomplete_details"), output
            ),
            output=list(output),
        )

    def event(self, e: dict[str, Any]) -> list[StreamEvent]:
        t = e.get("type")
        if t is None and isinstance(e.get("error"), dict):
            # OpenRouter's generic (Chat-Completions-shaped) mid-stream error frame.
            raise _error_from(e["error"].get("message"), e["error"].get("code"))
        match t:
            case "response.created" | "response.in_progress":
                r = e.get("response") or {}
                self.response_id = r.get("id") or self.response_id
                self.model = r.get("model") or self.model
                cls = (
                    ResponseCreatedEvent
                    if t == "response.created"
                    else ResponseInProgressEvent
                )
                return [cls(response=self._info(r))]
            case "response.output_item.added":
                raw = e.get("item") or {}
                self.partial[self._index(e)] = _Partial(raw=dict(raw))
                item = self._filtered(parse_item(raw))
                if item is None:
                    return []
                return [
                    ResponseOutputItemAddedEvent(output_index=self._index(e), item=item)
                ]
            case "response.output_item.done":
                self.content_seen = True
                item = parse_item(e.get("item") or {})
                if item is None:
                    return []
                i = self._index(e)
                prev = self.partial.get(i)
                # Some providers omit the call_id or name on `done`; keep the earlier one.
                if isinstance(item, FunctionCall) and prev:
                    item.call_id = prev.raw.get("call_id") or item.call_id
                    item.name = item.name or prev.raw.get("name") or ""
                self.done[i] = item
                item = self._filtered(item)
                if item is None:
                    return []
                return [ResponseOutputItemDoneEvent(output_index=i, item=item)]
            case "response.output_text.delta" | "response.content_part.delta":
                d = e.get("delta")
                if not isinstance(d, str) or not d:
                    return []
                self.content_seen = True
                p = self._partial(e, "message")
                p.text.append(d)
                return [
                    ResponseOutputTextDeltaEvent(
                        item_id=e.get("item_id") or p.raw.get("id") or "",
                        output_index=self._index(e),
                        content_index=e.get("content_index") or 0,
                        delta=d,
                    )
                ]
            case "response.function_call_arguments.delta":
                self.content_seen = True
                p = self._partial(e, "function_call")
                if isinstance(e.get("delta"), str):
                    p.args.append(e["delta"])
                return self._validate(ResponseFunctionCallArgumentsDeltaEvent, e)
            case "response.function_call_arguments.done":
                p = self._partial(e, "function_call")
                if isinstance(e.get("arguments"), str):
                    p.args = [e["arguments"]]
                return self._validate(ResponseFunctionCallArgumentsDoneEvent, e)
            case (
                "response.reasoning_text.delta"
                | "response.reasoning.delta"
                | "response.reasoning_summary_text.delta"
            ):
                if not self.reasoning_enabled or not isinstance(e.get("delta"), str):
                    return []
                p = self._partial(e, "reasoning")
                p.reasoning.append(e["delta"])
                if t == "response.reasoning_summary_text.delta":
                    return self._validate(ResponseReasoningSummaryTextDeltaEvent, e)
                return self._validate(
                    ResponseReasoningTextDeltaEvent,
                    {**e, "type": "response.reasoning_text.delta"},
                )
            case (
                "response.reasoning_text.done"
                | "response.reasoning_summary_text.done"
                | "response.reasoning_summary_part.added"
                | "response.reasoning_summary_part.done"
            ):
                if not self.reasoning_enabled:
                    return []
                cls2: Any = {
                    "response.reasoning_text.done": ResponseReasoningTextDoneEvent,
                    "response.reasoning_summary_text.done": ResponseReasoningSummaryTextDoneEvent,
                    "response.reasoning_summary_part.added": ResponseReasoningSummaryPartAddedEvent,
                    "response.reasoning_summary_part.done": ResponseReasoningSummaryPartDoneEvent,
                }[t]
                return self._validate(cls2, e)
            case "response.content_part.added" | "response.content_part.done":
                try:
                    part = _CONTENT.validate_python(e.get("part"))
                except ValidationError:
                    return []
                cls3 = (
                    ResponseContentPartAddedEvent
                    if t == "response.content_part.added"
                    else ResponseContentPartDoneEvent
                )
                return self._validate(cls3, {**e, "part": part})
            case "response.output_text.done":
                return self._validate(ResponseOutputTextDoneEvent, e)
            case "response.output_text.annotation.added":
                return self._validate(ResponseOutputTextAnnotationAddedEvent, e)
            case "response.refusal.delta":
                self.content_seen = True
                return self._validate(ResponseRefusalDeltaEvent, e)
            case "response.refusal.done":
                return self._validate(ResponseRefusalDoneEvent, e)
            case "response.completed" | "response.done" | "response.incomplete":
                return [
                    self.finish(e.get("response") or {}, t == "response.incomplete")
                ]
            case "response.failed":
                err = (e.get("response") or {}).get("error") or {}
                raise _error_from(
                    err.get("message") or "response failed", err.get("code")
                )
            case "error":
                inner = e.get("error")
                err2: dict[str, Any] = inner if isinstance(inner, dict) else e
                raise _error_from(err2.get("message"), err2.get("code"))
        return []

    def _assembled(self) -> list[Item]:
        out: list[Item] = []
        for i in sorted(set(self.partial) | set(self.done)):
            if i in self.done:
                out.append(self.done[i])
                continue
            p = self.partial[i]
            raw = dict(p.raw)
            match raw.get("type"):
                case "message":
                    if p.text:
                        raw["content"] = [
                            {"type": "output_text", "text": "".join(p.text)}
                        ]
                case "function_call":
                    if p.args:
                        raw["arguments"] = "".join(p.args)
                case "reasoning":
                    if p.reasoning:
                        raw["content"] = [
                            {"type": "reasoning_text", "text": "".join(p.reasoning)}
                        ]
            if item := parse_item(raw):
                out.append(item)
        return out

    def finish(
        self, r: dict[str, Any], incomplete: bool
    ) -> ResponseCompletedEvent | ResponseIncompleteEvent:
        self.finished = True
        final = [x for o in r.get("output") or [] if (x := parse_item(o))]
        items = final or self._assembled()
        items = [i for i in items if self._keep(i)]
        if not r.get("status"):
            r = {**r, "status": "incomplete" if incomplete else "completed"}
        info = self._info(r, items)
        if incomplete or r.get("status") == "incomplete":
            return ResponseIncompleteEvent(response=info)
        return ResponseCompletedEvent(response=info)

    def end(self) -> list[StreamEvent]:
        """The stream ended: synthesize a terminal event if the provider sent none."""
        if self.finished:
            return []
        if not self.content_seen and not self.partial:
            raise LLMError("empty stream", None, True)
        return [self.finish({}, False)]


def iter_sse(on_data: Callable[[str], None]) -> Callable[[str | None], None]:
    """An incremental SSE decoder: feed text chunks, then `None` at the end."""
    buf = [""]
    data: list[str] = []

    def dispatch() -> None:
        if data:
            payload = "\n".join(data)
            data.clear()
            if payload.strip() and payload.strip() != "[DONE]":
                on_data(payload)

    def feed(chunk: str | None) -> None:
        if chunk is None:
            if buf[0].strip():
                line = buf[0].removesuffix("\r")
                if line.startswith("data:"):
                    data.append(line[5:].lstrip())
            buf[0] = ""
            dispatch()
            return
        buf[0] += chunk
        while (nl := buf[0].find("\n")) >= 0:
            line = buf[0][:nl].removesuffix("\r")
            buf[0] = buf[0][nl + 1 :]
            if not line:
                dispatch()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
            # comments (": OPENROUTER PROCESSING"), event:, id:, retry: are ignored

    return feed


class OpenAIAPIProvider(Provider):
    """Base provider for services that speak the OpenAI Responses API."""

    def __init__(
        self,
        name: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
        headers: dict[str, str] | None = None,
    ):
        think = model.endswith(":think")
        model = model.removesuffix(":think")
        super().__init__(name=name, model=model)
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.client = client
        """Shared HTTP client; without one, each call opens and closes its own."""
        self.extra_headers: dict[str, str] = dict(headers or {})
        self.extra_body: dict[str, Any] = {}
        self._think_in_model_string = think

    @override
    def with_model(self, model: str) -> "OpenAIAPIProvider":
        p = copy.copy(self)
        p._model = model.removesuffix(":think")
        p._think_in_model_string = self._think_in_model_string or model.endswith(
            ":think"
        )
        p._Provider__context_length = None  # type: ignore[attr-defined]
        p.extra_headers = dict(self.extra_headers)
        p.extra_body = dict(self.extra_body)
        return p

    # ----- reasoning -----

    def reasoning_enabled(self, options: LLMOptions) -> bool:
        if (r := options.reasoning) and r.enabled is not None:
            return r.enabled
        if self._think_in_model_string:
            return True
        return _truthy(os.environ.get("AGENTIA_REASONING_ENABLED"))

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
        exclude = _truthy(os.environ.get("AGENTIA_REASONING_EXCLUDE"))
        x = os.environ.get("AGENTIA_REASONING_MAX_TOKENS")
        max_tokens = int(x) if x else None
        if (r := options.reasoning) and enabled:
            if r.effort is not None:
                effort = r.effort
            if r.exclude is not None:
                exclude = r.exclude
            if r.max_tokens is not None:
                max_tokens = r.max_tokens
        return self.get_reasoning_args(enabled, effort, exclude, max_tokens)

    # ----- request -----

    def default_cache_control(self) -> bool:
        """Whether to send explicit prompt-cache breakpoints when the options don't say."""
        return False

    def _instructions(
        self, request: ModelRequest
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """(`instructions`, leading system messages) for a request."""
        static = [
            i.text for i in request.instructions if not i.dynamic and i.text.strip()
        ]
        dynamic = [i.text for i in request.instructions if i.dynamic and i.text.strip()]
        summary = (
            request.summary if request.summary and request.summary.strip() else None
        )
        cache = request.options.cache_control
        if cache is None:
            cache = self.default_cache_control()
        if not cache:
            text = "\n\n".join([*static, *([summary] if summary else []), *dynamic])
            return (text or None), []
        bp = {"type": "ephemeral"}
        parts: list[dict[str, Any]] = []
        for k, t in enumerate(static):
            part: dict[str, Any] = {"type": "input_text", "text": t}
            if k == len(static) - 1:
                part["cache_control"] = bp
            parts.append(part)
        if summary:
            parts.append({"type": "input_text", "text": summary, "cache_control": bp})
        parts += [{"type": "input_text", "text": t} for t in dynamic]
        if not parts:
            return None, []
        return None, [{"type": "message", "role": "system", "content": parts}]

    def _tool_choice(self, tool_choice: ToolChoice | None) -> Any:
        match tool_choice:
            case None:
                return None
            case "none" | "auto" | "required":
                return tool_choice
            case _:
                return {"type": "function", "name": tool_choice}

    def build_body(self, request: ModelRequest, model: str) -> dict[str, Any]:
        options = request.options
        rf = options.response_format
        if rf and not isinstance(rf, (ResponseFormatJson, ResponseFormatText)):
            rf = ResponseFormatJson.from_model(rf)
        instructions, system = self._instructions(request)
        tools = [t.to_responses_tool() for t in request.tools]
        body: dict[str, Any] = {
            "model": model,
            "input": system + to_responses_input(request.items),
            "instructions": instructions,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "max_output_tokens": options.max_output_tokens,
            "text": {"format": rf.to_responses_text_format()} if rf else None,
            "tools": tools or None,
            "tool_choice": self._tool_choice(options.tool_choice) if tools else None,
            # Only meaningful alongside tools, and rejected by some providers without them.
            "parallel_tool_calls": options.parallel_tool_calls if tools else None,
            # Never persist conversations server-side; the history lives in the agent.
            "store": False,
            "stream": True,
        }
        if self.reasoning_enabled(options):
            # Without server-side storage, reasoning can only be replayed if the
            # provider hands back the encrypted payload.
            body["include"] = ["reasoning.encrypted_content"]
        body = {k: v for k, v in body.items() if v is not None}
        body.update(self.extra_body)
        if reasoning_args := self.get_reasoning_args_from_options(options):
            body["reasoning"] = reasoning_args
        # The Responses API has no first-class fields for these, so they are only
        # forwarded when explicitly set, for gateways that still accept them.
        for key, value in (
            ("frequency_penalty", options.frequency_penalty),
            ("presence_penalty", options.presence_penalty),
            ("seed", options.seed),
            ("stop", options.stop_sequences),
            ("top_k", options.top_k),
        ):
            if value is not None:
                body[key] = value
        if options.extra_body:
            body.update(options.extra_body)
        return body

    def headers(self) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}",
            **self.extra_headers,
        }

    # ----- streaming -----

    @override
    async def stream(
        self, request: ModelRequest, client: httpx.AsyncClient | None = None
    ) -> AsyncGenerator[StreamEvent, None]:
        options = request.options
        models: list[str] = list(
            dict.fromkeys([self.model, *(options.fallback_models or [])])
        )
        retries = (
            DEFAULT_MAX_RETRIES if options.max_retries is None else options.max_retries
        )
        last: LLMError | None = None
        for model in models:
            for attempt in range(retries + 1):
                committed = [False]
                try:
                    async for event in self._stream_once(
                        request, model, client, committed
                    ):
                        yield event
                    return
                except LLMError as e:
                    last = e
                except httpx.HTTPError as e:
                    last = LLMError(f"{type(e).__name__}: {e}", None, True)
                # Once output has reached the caller, a retry would show it twice.
                if committed[0]:
                    raise last
                if not last.retryable:
                    break
                if attempt < retries:
                    await asyncio.sleep(
                        min(20.0, 0.5 * 2**attempt) * (0.5 + random.random())
                    )
            # Auth and billing errors won't be fixed by another model; anything else
            # (unknown model, unsupported parameter, exhausted retries) may be.
            if last and last.status in (401, 402, 403):
                break
        assert last is not None
        raise last

    async def _stream_once(
        self,
        request: ModelRequest,
        model: str,
        client: httpx.AsyncClient | None,
        committed: list[bool],
    ) -> AsyncGenerator[StreamEvent, None]:
        body = self.build_body(request, model)
        acc = StreamAccumulator(self.reasoning_enabled(request.options))
        http = client or self.client
        own = http is None
        if http is None:
            http = httpx.AsyncClient()
        timeout = httpx.Timeout(60.0, read=request.options.timeout or 600.0)
        try:
            async with http.stream(
                "POST",
                f"{self.base_url}/responses",
                headers=self.headers(),
                json=body,
                timeout=timeout,
            ) as res:
                if res.status_code >= 400:
                    text = (await res.aread()).decode(errors="replace")
                    code = res.status_code
                    raise LLMError(
                        f"LLM HTTP {code}: {text[:500]}",
                        code,
                        code in (408, 429) or code >= 500,
                    )
                if "application/json" in res.headers.get("content-type", ""):
                    # Some proxies ignore `stream: true` and return the Response object.
                    r = json.loads(await res.aread())
                    if isinstance(r.get("error"), dict):
                        raise _error_from(
                            r["error"].get("message"), r["error"].get("code")
                        )
                    committed[0] = True
                    yield acc.finish(r, r.get("status") == "incomplete")
                    return
                payloads: list[str] = []
                feed = iter_sse(payloads.append)

                async for chunk in res.aiter_text():
                    feed(chunk)
                    # One payload at a time: events before an error frame reach the
                    # caller before the error is raised.
                    while payloads:
                        for ev in _decode(acc, payloads.pop(0)):
                            if ev.type not in (
                                "response.created",
                                "response.in_progress",
                            ):
                                committed[0] = True
                            yield ev
                feed(None)
                while payloads:
                    for ev in _decode(acc, payloads.pop(0)):
                        committed[0] = True
                        yield ev
                for ev in acc.end():
                    committed[0] = True
                    yield ev
        finally:
            if own:
                await http.aclose()

    # ----- model metadata -----

    async def _models_catalog(
        self, url: str, auth: bool = True, refresh: bool = False
    ) -> dict[str, int]:
        hit = _models_cache.get(url)
        if hit and not refresh and time.monotonic() - hit[0] < _MODELS_TTL_S:
            return hit[1]
        http = self.client or httpx.AsyncClient()
        try:
            res = await http.get(
                url, headers={"authorization": f"Bearer {self.api_key}"} if auth else {}
            )
            res.raise_for_status()
            out: dict[str, int] = {}
            for m in res.json().get("data", []):
                mid = m.get("id")
                ctx = m.get("context_length") or m.get("context_window")
                if mid and ctx is not None:
                    out[mid] = int(ctx)
            _models_cache[url] = (time.monotonic(), out)
            return out
        finally:
            if http is not self.client:
                await http.aclose()

    async def _context_length_from(
        self, url: str, model: str, auth: bool = True
    ) -> int:
        catalog = await self._models_catalog(url, auth)
        if model not in catalog:
            # A new model, or a stale cache: fetch once more.
            catalog = await self._models_catalog(url, auth, refresh=True)
        if model in catalog:
            return catalog[model]
        raise ValueError(f"Context length not available for model '{model}' from {url}")

    @override
    async def _fetch_context_length(self) -> int:
        return await self._context_length_from(f"{self.base_url}/models", self.model)


def _decode(acc: StreamAccumulator, data: str) -> list[StreamEvent]:
    try:
        e = json.loads(data)
    except ValueError:
        return []
    return acc.event(e) if isinstance(e, dict) else []


__all__ = [
    "OpenAIAPIProvider",
    "StreamAccumulator",
    "finish_reason_of",
    "iter_sse",
    "parse_item",
    "parse_usage",
]
