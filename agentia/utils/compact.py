from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

from agentia.models import Message, ToolMessage

if TYPE_CHECKING:
    from agentia.agent import Agent

EFFORT_INSTRUCTIONS = {
    "low": "Keep as much important information as possible. Only drop clearly unimportant or redundant content.",
    "medium": "Keep important information and context, but drop routine details, verbose outputs, and redundant exchanges.",
    "high": "Be very aggressive in compacting. Only keep the most critical information and drop everything else.",
}


def _format_messages(summary: str | None, messages: Sequence[Message]) -> str:
    lines: list[str] = []
    if summary:
        lines.append(f"[summary of the earlier conversation]: {summary}")
    for msg in messages:
        if not isinstance(msg, ToolMessage) and msg.text:
            lines.append(f"[{msg.role}]: {msg.text}")
    return "\n\n".join(lines)


async def compact_history(
    agent: Agent,
    effort: Literal["low", "medium", "high"] = "low",
    model: str | None = None,
    keep_last_n: int = 1,
) -> None:
    from agentia.agent import Agent

    messages = agent.history.get()
    split = len(messages) - keep_last_n
    # Never separate tool results from the assistant message that requested them.
    while split > 0 and isinstance(messages[split], ToolMessage):
        split -= 1
    if split <= 0:
        return

    compact_instructions = (
        "Below is a conversation history. Your task is to compact it into a single concise summary "
        "that preserves all important information (key decisions, facts, code changes, conclusions, "
        "user preferences, and action items) while dropping unimportant details.\n\n"
        f"{EFFORT_INSTRUCTIONS[effort]}\n\n"
        "Output ONLY the compacted summary, nothing else."
    )

    to_compact = messages[:split]
    conversation_text = _format_messages(agent.history.summary, to_compact)

    compact_agent = Agent(model=model or agent.model, instructions=compact_instructions)
    result = await compact_agent.run(conversation_text)

    agent.history.compact(result.text, len(to_compact))
