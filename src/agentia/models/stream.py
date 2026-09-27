"""Streaming events.

Event `type` strings are exactly the ones the Responses API emits, so consumers can match
on them directly. Every event carries a `sequence_number`; item-scoped events also carry
`item_id`, `output_index`, and `content_index` or `summary_index`.

Responses API types
-------------------
Lifecycle of one response:
- `ResponseCreatedEvent`                      -- `response.created`
- `ResponseInProgressEvent`                   -- `response.in_progress`
- `ResponseCompletedEvent`                    -- `response.completed`
- `ResponseIncompleteEvent`                   -- `response.incomplete`
- `ResponseFailedEvent`                       -- `response.failed`

Output items and content parts:
- `ResponseOutputItemAddedEvent`              -- `response.output_item.added`
- `ResponseOutputItemDoneEvent`               -- `response.output_item.done`
- `ResponseContentPartAddedEvent`             -- `response.content_part.added`
- `ResponseContentPartDoneEvent`              -- `response.content_part.done`

Assistant text:
- `ResponseOutputTextDeltaEvent`              -- `response.output_text.delta`
- `ResponseOutputTextDoneEvent`               -- `response.output_text.done`
- `ResponseOutputTextAnnotationAddedEvent`    -- `response.output_text.annotation.added`
- `ResponseRefusalDeltaEvent`                 -- `response.refusal.delta`
- `ResponseRefusalDoneEvent`                  -- `response.refusal.done`

Reasoning:
- `ResponseReasoningTextDeltaEvent`           -- `response.reasoning_text.delta`
- `ResponseReasoningTextDoneEvent`            -- `response.reasoning_text.done`
- `ResponseReasoningSummaryTextDeltaEvent`    -- `response.reasoning_summary_text.delta`
- `ResponseReasoningSummaryTextDoneEvent`     -- `response.reasoning_summary_text.done`
- `ResponseReasoningSummaryPartAddedEvent`    -- `response.reasoning_summary_part.added`
- `ResponseReasoningSummaryPartDoneEvent`     -- `response.reasoning_summary_part.done`

Tool calls and errors:
- `ResponseFunctionCallArgumentsDeltaEvent`   -- `response.function_call_arguments.delta`
- `ResponseFunctionCallArgumentsDoneEvent`    -- `response.function_call_arguments.done`
- `ErrorEvent`                                -- `error`
- `ResponseStatus`                            -- the `response.status` values

`response.output_item.*` covers only what the model produced -- `message`, `reasoning`,
`function_call`, and provider-executed built-in tool calls, whose results the server
streams because the server ran them.

Agentia extensions
------------------
- `RunStartedEvent`    -- `agentia.run.started`. One `Agent.run()` spans several
                          responses as the ReAct loop iterates, which the API has no
                          event for.
- `RunCompletedEvent`  -- `agentia.run.completed`, carrying the run's total `usage` and
                          `finish_reason`.
- `InputItemAddedEvent` -- `agentia.input_item.added`, for items agentia appends itself:
                          a client-executed tool result, or an item from `Agent.defer()`.
                          A `function_call_output` is an *input* to the next request, so
                          the API has no event for it and it is never a
                          `response.output_item.*`.
- `ResponseInfo`       -- the subset of the API's `response` object that agentia
                          surfaces, with a normalized `finish_reason` added.
- `StreamEvent`        -- the union of every event above.

Events this build does not emit: the API also defines `response.queued`, audio events,
and per-built-in-tool progress events (`response.web_search_call.*`,
`response.file_search_call.*`, `response.mcp_call.*`, `response.code_interpreter_call.*`,
`response.image_generation_call.*`, `response.custom_tool_call_input.*`).
"""

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from .base import Annotation, FinishReason, Usage
from .content import Content, SummaryText
from .items import Item

type ResponseStatus = Literal[
    "queued", "in_progress", "completed", "incomplete", "failed", "cancelled"
]


class ResponseInfo(BaseModel):
    """The subset of a Responses API `response` object that agentia surfaces."""

    id: str | None = None
    status: ResponseStatus | None = None
    usage: Usage = Usage()
    finish_reason: FinishReason = "unknown"
    output: Sequence[Item] = Field(default_factory=list)


class _Event(BaseModel):
    sequence_number: int = 0


# --- Response lifecycle ------------------------------------------------------


class ResponseCreatedEvent(_Event):
    type: Literal["response.created"] = "response.created"
    response: ResponseInfo


class ResponseInProgressEvent(_Event):
    type: Literal["response.in_progress"] = "response.in_progress"
    response: ResponseInfo


class ResponseCompletedEvent(_Event):
    type: Literal["response.completed"] = "response.completed"
    response: ResponseInfo


class ResponseIncompleteEvent(_Event):
    type: Literal["response.incomplete"] = "response.incomplete"
    response: ResponseInfo


class ResponseFailedEvent(_Event):
    type: Literal["response.failed"] = "response.failed"
    response: ResponseInfo


# --- Items -------------------------------------------------------------------


class ResponseOutputItemAddedEvent(_Event):
    type: Literal["response.output_item.added"] = "response.output_item.added"
    output_index: int = 0
    item: Item


class ResponseOutputItemDoneEvent(_Event):
    type: Literal["response.output_item.done"] = "response.output_item.done"
    output_index: int = 0
    item: Item


# --- Content parts -----------------------------------------------------------


class ResponseContentPartAddedEvent(_Event):
    type: Literal["response.content_part.added"] = "response.content_part.added"
    item_id: str
    output_index: int = 0
    content_index: int = 0
    part: Content


class ResponseContentPartDoneEvent(_Event):
    type: Literal["response.content_part.done"] = "response.content_part.done"
    item_id: str
    output_index: int = 0
    content_index: int = 0
    part: Content


# --- Output text -------------------------------------------------------------


class ResponseOutputTextDeltaEvent(_Event):
    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    item_id: str
    output_index: int = 0
    content_index: int = 0
    delta: str


class ResponseOutputTextDoneEvent(_Event):
    type: Literal["response.output_text.done"] = "response.output_text.done"
    item_id: str
    output_index: int = 0
    content_index: int = 0
    text: str = ""


class ResponseOutputTextAnnotationAddedEvent(_Event):
    type: Literal["response.output_text.annotation.added"] = (
        "response.output_text.annotation.added"
    )
    item_id: str
    output_index: int = 0
    content_index: int = 0
    annotation_index: int = 0
    annotation: Annotation


# --- Refusal -----------------------------------------------------------------


class ResponseRefusalDeltaEvent(_Event):
    type: Literal["response.refusal.delta"] = "response.refusal.delta"
    item_id: str
    output_index: int = 0
    content_index: int = 0
    delta: str


class ResponseRefusalDoneEvent(_Event):
    type: Literal["response.refusal.done"] = "response.refusal.done"
    item_id: str
    output_index: int = 0
    content_index: int = 0
    refusal: str = ""


# --- Reasoning ---------------------------------------------------------------


class ResponseReasoningTextDeltaEvent(_Event):
    type: Literal["response.reasoning_text.delta"] = "response.reasoning_text.delta"
    item_id: str
    output_index: int = 0
    content_index: int = 0
    delta: str


class ResponseReasoningTextDoneEvent(_Event):
    type: Literal["response.reasoning_text.done"] = "response.reasoning_text.done"
    item_id: str
    output_index: int = 0
    content_index: int = 0
    text: str = ""


class ResponseReasoningSummaryTextDeltaEvent(_Event):
    type: Literal["response.reasoning_summary_text.delta"] = (
        "response.reasoning_summary_text.delta"
    )
    item_id: str
    output_index: int = 0
    summary_index: int = 0
    delta: str


class ResponseReasoningSummaryTextDoneEvent(_Event):
    type: Literal["response.reasoning_summary_text.done"] = (
        "response.reasoning_summary_text.done"
    )
    item_id: str
    output_index: int = 0
    summary_index: int = 0
    text: str = ""


class ResponseReasoningSummaryPartAddedEvent(_Event):
    type: Literal["response.reasoning_summary_part.added"] = (
        "response.reasoning_summary_part.added"
    )
    item_id: str
    output_index: int = 0
    summary_index: int = 0
    part: SummaryText


class ResponseReasoningSummaryPartDoneEvent(_Event):
    type: Literal["response.reasoning_summary_part.done"] = (
        "response.reasoning_summary_part.done"
    )
    item_id: str
    output_index: int = 0
    summary_index: int = 0
    part: SummaryText


# --- Function calls ----------------------------------------------------------


class ResponseFunctionCallArgumentsDeltaEvent(_Event):
    type: Literal["response.function_call_arguments.delta"] = (
        "response.function_call_arguments.delta"
    )
    item_id: str
    output_index: int = 0
    delta: str


class ResponseFunctionCallArgumentsDoneEvent(_Event):
    type: Literal["response.function_call_arguments.done"] = (
        "response.function_call_arguments.done"
    )
    item_id: str
    output_index: int = 0
    arguments: str = ""


# --- Errors ------------------------------------------------------------------


class ErrorEvent(_Event):
    type: Literal["error"] = "error"
    message: str
    code: str | None = None
    param: str | None = None


# --- Agentia extensions ------------------------------------------------------


class RunStartedEvent(_Event):
    """A whole `Agent.run()` started. Agentia extension."""

    type: Literal["agentia.run.started"] = "agentia.run.started"


class RunCompletedEvent(_Event):
    """A whole `Agent.run()` finished, after any number of responses. Agentia extension."""

    type: Literal["agentia.run.completed"] = "agentia.run.completed"
    usage: Usage = Usage()
    finish_reason: FinishReason = "stop"


class InputItemAddedEvent(_Event):
    """Agentia appended an input item to the conversation. Agentia extension.

    Emitted for client-executed tool results (`function_call_output`) and for items
    enqueued with `Agent.defer()`. The Responses API has no event for these, because
    they are inputs to the next request rather than model output.
    """

    type: Literal["agentia.input_item.added"] = "agentia.input_item.added"
    item: Item


type StreamEvent = Annotated[
    ResponseCreatedEvent
    | ResponseInProgressEvent
    | ResponseCompletedEvent
    | ResponseIncompleteEvent
    | ResponseFailedEvent
    | ResponseOutputItemAddedEvent
    | ResponseOutputItemDoneEvent
    | ResponseContentPartAddedEvent
    | ResponseContentPartDoneEvent
    | ResponseOutputTextDeltaEvent
    | ResponseOutputTextDoneEvent
    | ResponseOutputTextAnnotationAddedEvent
    | ResponseRefusalDeltaEvent
    | ResponseRefusalDoneEvent
    | ResponseReasoningTextDeltaEvent
    | ResponseReasoningTextDoneEvent
    | ResponseReasoningSummaryTextDeltaEvent
    | ResponseReasoningSummaryTextDoneEvent
    | ResponseReasoningSummaryPartAddedEvent
    | ResponseReasoningSummaryPartDoneEvent
    | ResponseFunctionCallArgumentsDeltaEvent
    | ResponseFunctionCallArgumentsDoneEvent
    | ErrorEvent
    | RunStartedEvent
    | RunCompletedEvent
    | InputItemAddedEvent,
    Field(discriminator="type"),
]

__all__ = [
    "ErrorEvent",
    "InputItemAddedEvent",
    "ResponseCompletedEvent",
    "ResponseContentPartAddedEvent",
    "ResponseContentPartDoneEvent",
    "ResponseCreatedEvent",
    "ResponseFailedEvent",
    "ResponseFunctionCallArgumentsDeltaEvent",
    "ResponseFunctionCallArgumentsDoneEvent",
    "ResponseInProgressEvent",
    "ResponseIncompleteEvent",
    "ResponseInfo",
    "ResponseOutputItemAddedEvent",
    "ResponseOutputItemDoneEvent",
    "ResponseOutputTextAnnotationAddedEvent",
    "ResponseOutputTextDeltaEvent",
    "ResponseOutputTextDoneEvent",
    "ResponseReasoningSummaryPartAddedEvent",
    "ResponseReasoningSummaryPartDoneEvent",
    "ResponseReasoningSummaryTextDeltaEvent",
    "ResponseReasoningSummaryTextDoneEvent",
    "ResponseReasoningTextDeltaEvent",
    "ResponseReasoningTextDoneEvent",
    "ResponseRefusalDeltaEvent",
    "ResponseRefusalDoneEvent",
    "ResponseStatus",
    "RunCompletedEvent",
    "RunStartedEvent",
    "StreamEvent",
]
