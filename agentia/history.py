from typing import Callable, Literal, Sequence, overload
from agentia.spec import *


class History:
    def __init__(self) -> None:
        self.__instruction_generators: list[str | Callable[[], str | None]] = []
        self.__non_instruction_messages: list[NonSystemMessage] = []
        self.__live_cursor: int = 0
        self.current_tokens: int = 0
        self.usage: Usage = Usage()

    def add_instructions(self, instructions: str | Callable[[], str | None]) -> None:
        self.__instruction_generators.append(instructions)

    def get_instructions(self) -> str:
        return self.__get_instructions()

    def __get_instructions(self) -> str:
        instructions = []
        for gen in self.__instruction_generators:
            if callable(gen):
                result = gen()
                if result:
                    instructions.append(result)
            else:
                if gen:
                    instructions.append(gen)
        return "\n\n".join(instructions)

    def add(self, *messages: NonSystemMessage) -> None:
        self.__non_instruction_messages.extend(messages)

    @overload
    def get(self, include_instructions: Literal[True] = True) -> list[Message]: ...

    @overload
    def get(self, include_instructions: Literal[False]) -> list[NonSystemMessage]: ...

    def get(
        self, include_instructions: bool = True
    ) -> list[Message] | list[NonSystemMessage]:
        messages: list[Message] = []
        if include_instructions:
            if i := self.__get_instructions():
                messages.append(SystemMessage(content=i))
        messages.extend(self.__non_instruction_messages)
        return messages

    def get_new(self) -> Sequence[Message]:
        """Return only messages added since the last cursor advance."""
        return self.__non_instruction_messages[self.__live_cursor :]

    def advance_cursor(self) -> None:
        """Move cursor to the end of the current message list."""
        self.__live_cursor = len(self.__non_instruction_messages)

    def clear(self, clear_instructions: bool = False) -> None:
        self.__non_instruction_messages = []
        if clear_instructions:
            self.__instruction_generators = []
        self.__live_cursor = 0
        self.current_tokens = 0
        self.usage = Usage()

    def load(
        self, messages: Sequence[Message], include_instructions: bool = False
    ) -> None:
        self.clear(clear_instructions=include_instructions)
        for m in messages:
            if isinstance(m, SystemMessage):
                if include_instructions:
                    self.__instruction_generators.append(lambda m=m: m.content)
            else:
                self.__non_instruction_messages.append(m)


__all__ = ["History"]
