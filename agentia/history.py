from typing import Callable, Sequence
from agentia.models import *


class History:
    """
    The conversation history of an agent.

    After compaction, the history keeps a `summary` of the compacted messages. The
    model sees the instructions (with the summary appended) followed by the remaining
    messages. Compacted messages are not lost: `get_all()` still returns them.
    """

    def __init__(self) -> None:
        self.__instruction_generators: list[str | Callable[[], str | None]] = []
        self.__messages: list[Message] = []
        self.__compacted: list[Message] = []
        self.__summary: str | None = None
        self.__live_cursor: int = 0
        self.current_tokens: int = 0
        self.usage: Usage = Usage()

    def add_instructions(self, instructions: str | Callable[[], str | None]) -> None:
        self.__instruction_generators.append(instructions)

    def get_instructions(self) -> str:
        """The system prompt: all instructions, followed by the compaction summary."""
        instructions = []
        for gen in self.__instruction_generators:
            if callable(gen):
                result = gen()
                if result:
                    instructions.append(result)
            else:
                if gen:
                    instructions.append(gen)
        if self.__summary:
            instructions.append(
                "[[COMPACTED CONVERSATION SUMMARY]]\n"
                "The earlier part of this conversation was compacted into this summary:\n\n"
                f"{self.__summary}\n"
                "[[END OF CONVERSATION SUMMARY]]"
            )
        return "\n\n".join(instructions)

    @property
    def summary(self) -> str | None:
        """The summary of the compacted messages, if the history has been compacted."""
        return self.__summary

    def add(self, *messages: Message) -> None:
        self.__messages.extend(messages)

    def get(self) -> list[Message]:
        """The messages sent to the model (i.e. excluding compacted messages)."""
        return list(self.__messages)

    def get_all(self) -> list[Message]:
        """All messages, including the ones replaced by the compaction summary."""
        return self.__compacted + self.__messages

    def get_new(self) -> Sequence[Message]:
        """Return only messages added since the last cursor advance."""
        return self.__messages[self.__live_cursor :]

    def advance_cursor(self) -> None:
        """Move cursor to the end of the current message list."""
        self.__live_cursor = len(self.__messages)

    def compact(self, summary: str, count: int) -> None:
        """Replace the first `count` messages with `summary` (which must also cover any previous summary)."""
        count = min(count, len(self.__messages))
        self.__compacted.extend(self.__messages[:count])
        self.__messages = self.__messages[count:]
        self.__live_cursor = max(0, self.__live_cursor - count)
        self.__summary = summary

    def clear(self) -> None:
        self.__messages = []
        self.__compacted = []
        self.__summary = None
        self.__live_cursor = 0
        self.current_tokens = 0
        self.usage = Usage()

    def load(self, messages: Sequence[Message], summary: str | None = None) -> None:
        self.clear()
        self.__messages.extend(messages)
        self.__summary = summary


__all__ = ["History"]
