from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

from agentia.spec import AssistantMessage, Message
from agentia.spec.chat import MessagePartText, MessagePartReasoning

if TYPE_CHECKING:
    from agentia.agent import Agent

EFFORT_INSTRUCTIONS = {
    "low": "Keep as much important information as possible. Only drop clearly unimportant or redundant content.",
    "medium": "Keep important information and context, but drop routine details, verbose outputs, and redundant exchanges.",
    "high": "Be very aggressive in compacting. Only keep the most critical information and drop everything else.",
}


def _format_messages(messages: Sequence[Message]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = msg.role
        if isinstance(msg.content, str):
            lines.append(f"[{role}]: {msg.content}")
        elif isinstance(msg.content, Sequence):
            parts = []
            for part in msg.content:
                if isinstance(part, MessagePartText):
                    parts.append(part.text)
            if parts:
                lines.append(f"[{role}]: {''.join(parts)}")
    return "\n\n".join(lines)


async def compact_history(
    agent: Agent,
    effort: Literal["low", "medium", "high"] = "low",
    model: str | None = None,
) -> None:
    from agentia.agent import Agent

    non_instruction_messages = agent.history.get(include_instructions=False)
    if not non_instruction_messages:
        return
    length = len(non_instruction_messages)

    compact_instructions = (
        "Below is a conversation history. Your task is to compact it into a single concise summary "
        "that preserves all important information (key decisions, facts, code changes, conclusions, "
        "user preferences, and action items) while dropping unimportant details.\n\n"
        f"{EFFORT_INSTRUCTIONS[effort]}\n\n"
        "Output ONLY the compacted summary, nothing else."
    )

    conversation_text = _format_messages(non_instruction_messages)

    compact_agent = Agent(model=model or agent.model, instructions=compact_instructions)
    result = await compact_agent.run(conversation_text)

    agent.history.clear(clear_instructions=False)
    agent.history.add(
        AssistantMessage(
            content=[
                MessagePartText(
                    text=f"[[COMPACTED CONVERSATION SUMMARY FROM {length} MESSAGES]]"
                ),
                MessagePartText(text=result.text),
                MessagePartText(text="[[END OF CONVERSATION SUMMARY]]"),
            ]
        )
    )
