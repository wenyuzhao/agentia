from typing import Callable, Sequence
from agentia.spec import *


class History:
    def __init__(self) -> None:
        self.__instruction_generators: list[str | Callable[[], str | None]] = []
        self.__non_instruction_messages: list[
            UserMessage | AssistantMessage | ToolMessage
        ] = []

    def add_instructions(self, instructions: str | Callable[[], str | None]) -> None:
        self.__instruction_generators.append(instructions)

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

    def add(self, *messages: UserMessage | AssistantMessage | ToolMessage) -> None:
        self.__non_instruction_messages.extend(messages)

    def get(self, include_instructions: bool = True) -> list[Message]:
        messages: list[Message] = []
        if include_instructions:
            if i := self.__get_instructions():
                messages.append(SystemMessage(content=i))
        messages.extend(self.__non_instruction_messages)
        return messages

    def clear(self, clear_instructions: bool = False) -> None:
        self.__non_instruction_messages = []
        if clear_instructions:
            self.__instruction_generators = []

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
