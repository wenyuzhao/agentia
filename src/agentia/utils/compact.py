from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from agentia.models import (
    Compaction,
    FunctionCall,
    FunctionCallOutput,
    Item,
    Message,
    Reasoning,
)

if TYPE_CHECKING:
    from agentia.agent import Agent

EFFORT_INSTRUCTIONS = {
    "low": "Keep as much important information as possible. Only drop clearly unimportant or redundant content.",
    "medium": "Keep important information and context, but drop routine details, verbose outputs, and redundant exchanges.",
    "high": "Be very aggressive in compacting. Only keep the most critical information and drop everything else.",
}


def _format_items(items: Sequence[Item]) -> str:
    lines: list[str] = []
    for item in items:
        match item:
            case Message():
                if text := item.text:
                    lines.append(f"[{item.role}]: {text}")
            case FunctionCall():
                lines.append(f"[tool call {item.name}]: {item.arguments}")
            case FunctionCallOutput():
                lines.append(f"[tool result {item.name or item.call_id}]: {item.text}")
            case Reasoning():
                if text := item.display_text:
                    lines.append(f"[reasoning]: {text}")
            case Compaction():
                lines.append(f"[summary]: {item.summary}")
    return "\n\n".join(lines)


async def compact_history(
    agent: Agent,
    effort: Literal["low", "medium", "high"] = "low",
    model: str | None = None,
    keep_last_n: int = 1,
) -> None:
    from agentia.agent import Agent

    items = agent.history.get()
    length = len(items)
    if length <= keep_last_n:
        return

    compact_instructions = (
        "Below is a conversation history. Your task is to compact it into a single concise summary "
        "that preserves all important information (key decisions, facts, code changes, conclusions, "
        "user preferences, and action items) while dropping unimportant details.\n\n"
        f"{EFFORT_INSTRUCTIONS[effort]}\n\n"
        "Output ONLY the compacted summary, nothing else."
    )

    conversation_text = _format_items(items[:-keep_last_n])

    compact_agent = Agent(model=model or agent.model, instructions=compact_instructions)
    result = await compact_agent.run(conversation_text)

    items_to_keep = items[-keep_last_n:]
    agent.history.clear()
    agent.history.add(
        Compaction(summary=result.text, original_items=list(items)),
        *items_to_keep,
    )
