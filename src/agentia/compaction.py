"""Client-side compaction.

When the context grows past a threshold, older items are folded (together with the
previous summary) into a new plain-text summary. The history is never rewritten: a
`Compaction` records how many items it covers, and the model then sees the summary plus
the items after it.

The split keeps the last `keep_exchanges` user↔assistant exchanges verbatim and only
cuts before a user message or before a model response, so a function call is never
separated from its output. Inside one very long exchange it keeps the last
`keep_steps` model responses.
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel

from agentia.models import (
    Compaction,
    FunctionCall,
    FunctionCallOutput,
    Item,
    Message,
    Reasoning,
    ResponseCompletedEvent,
    ResponseIncompleteEvent,
    Usage,
)

if TYPE_CHECKING:
    from agentia.agent import Agent

DEFAULT_INSTRUCTIONS = """You are compacting a long-running conversation between a user and an AI agent so the agent can continue with a bounded context. Produce an updated summary that replaces the previous summary and the old events below.

Keep: the user's instructions and preferences (verbatim), the current task and its state, decisions and facts, file paths, IDs, URLs, commands and error messages (exact), and open items. Drop routine details, verbose tool outputs and redundant exchanges. Never copy secrets. Output only the summary."""


class CompactionOptions(BaseModel):
    auto: bool = True
    """Compact automatically before a model call when the context is over the threshold."""

    threshold: float = 0.6
    """Fraction of the model's context window that triggers compaction."""

    max_threshold_tokens: int | None = 100_000
    """Upper bound on the threshold, in tokens (None: no cap)."""

    keep_exchanges: int = 4
    keep_steps: int = 6
    model: str | None = None
    """Model for the summarizer (default: the agent's model)."""

    instructions: str = DEFAULT_INSTRUCTIONS
    max_output_tokens: int | None = 6000

    def threshold_tokens(self, context_window: int) -> int:
        t = int(context_window * self.threshold)
        return min(t, self.max_threshold_tokens) if self.max_threshold_tokens else t


def estimate_tokens(text: str) -> int:
    """A rough token count (~4 characters per token)."""
    return -(-len(text) // 4)


def item_text(item: Item) -> str:
    match item:
        case Message():
            return item.text
        case FunctionCall():
            return item.name + item.arguments
        case FunctionCallOutput():
            return item.text
        case Reasoning():
            return item.display_text
    return ""


def is_user_turn(item: Item) -> bool:
    return isinstance(item, Message) and item.role == "user"


def is_model_output(item: Item) -> bool:
    return isinstance(item, (FunctionCall, Reasoning)) or (
        isinstance(item, Message) and item.role == "assistant"
    )


def split_point(
    items: Sequence[Item],
    keep_exchanges: int = 4,
    keep_steps: int = 6,
    is_turn_start: Callable[[Item], bool] = is_user_turn,
) -> int:
    """How many leading items to fold into the summary (0 = nothing to compact).
    `is_turn_start` tells where an exchange starts (default: any user message)."""
    if keep_exchanges <= 0:
        return len(items)
    starts = [i for i, e in enumerate(items) if is_turn_start(e)]
    if len(starts) > keep_exchanges:
        return starts[len(starts) - keep_exchanges]
    responses = [
        i
        for i, e in enumerate(items)
        if is_model_output(e) and not (i > 0 and is_model_output(items[i - 1]))
    ]
    if len(responses) > keep_steps * 2:
        return responses[len(responses) - keep_steps]
    return 0


def render_items(items: Sequence[Item]) -> str:
    """Items rendered compactly for the summarizer."""
    out: list[str] = []
    for e in items:
        match e:
            case Message() if e.role == "assistant":
                if e.text.strip():
                    out.append(f"### Assistant\n{e.text}")
            case Message():
                out.append(f"### {e.role.capitalize()}\n{e.text}")
            case FunctionCall():
                out.append(f"### Tool call\n- call {e.name} {e.arguments[:300]}")
            case FunctionCallOutput():
                t = e.text
                body = t[:1200] + "…" if len(t) > 1200 else t
                out.append(f"### Tool result ({e.name or e.call_id})\n{body}")
    return "\n\n".join(out)


def compaction_input(
    previous: Compaction | None, rendered: str, focus: str | None = None
) -> str:
    parts = [
        f"# Previous summary\n\n{previous.summary}"
        if previous
        else "# Previous summary\n\n(none — this is the first compaction)",
        f"# Old events to fold in\n\n{rendered}",
    ]
    if focus:
        parts.append(f"# Emphasis requested by the user\n{focus}")
    return "\n\n---\n\n".join(parts)


async def compact(
    agent: "Agent",
    *,
    focus: str | None = None,
    keep_exchanges: int | None = None,
    options: CompactionOptions | None = None,
) -> Compaction | None:
    """Fold old items into a new summary and commit it. Returns None if there is
    nothing old enough to compact. `keep_exchanges=0` folds everything."""
    from agentia.llm.providers import ModelRequest
    from agentia.models import Instruction

    opts = options or agent.compaction or CompactionOptions()
    history = agent.history
    previous = history.compaction
    context = history.context()
    keep = opts.keep_exchanges if keep_exchanges is None else keep_exchanges
    cut = split_point(context, keep, opts.keep_steps, agent.is_turn_start)
    if cut <= 0:
        return None
    old = context[:cut]
    text = await agent.render_compaction_input(previous, old, focus)
    provider = agent.provider.with_model(opts.model) if opts.model else agent.provider
    request = ModelRequest(
        items=[Message.user(text)],
        instructions=[Instruction(text=opts.instructions)],
        options=agent.options.model_copy(
            update={
                "max_output_tokens": opts.max_output_tokens,
                "response_format": None,
                "tool_choice": None,
                "reasoning": None,
                "max_steps": None,
                "cache_control": False,
            }
        ),
    )
    summary = ""
    usage = Usage()
    model = provider.model
    async for event in provider.stream(request, agent.http_client):
        if isinstance(event, (ResponseCompletedEvent, ResponseIncompleteEvent)):
            usage = event.response.usage
            model = event.response.model or model
            summary = "".join(
                i.text
                for i in event.response.output
                if isinstance(i, Message) and i.role == "assistant"
            )
    summary = summary.strip()
    if not summary:
        raise RuntimeError("the summarizer returned nothing")
    c = Compaction(
        summary=summary,
        covers=(previous.covers if previous else 0) + cut,
        model=model,
        usage=usage,
    )
    # The host may persist transcripts, record cost, or veto (by raising) first.
    await agent.on_compaction(c, old)
    history.add_compaction(c)
    history.usage += usage
    return c


def context_tokens(
    instructions: Sequence[str], summary: str | None, items: Sequence[Item]
) -> int:
    """Estimated prompt size of a request, ignoring images and tool schemas."""
    chars = sum(len(t) for t in instructions) + len(summary or "")
    chars += sum(len(item_text(i)) for i in items)
    return -(-chars // 4)


__all__ = [
    "DEFAULT_INSTRUCTIONS",
    "CompactionOptions",
    "compact",
    "compaction_input",
    "context_tokens",
    "estimate_tokens",
    "render_items",
    "split_point",
]
