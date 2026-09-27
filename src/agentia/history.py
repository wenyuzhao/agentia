from typing import Callable, Sequence
from agentia.models import *


class History:
    def __init__(self) -> None:
        self.__instruction_generators: list[str | Callable[[], str | None]] = []
        self.__items: list[Item] = []
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

    def add(self, *items: Item) -> None:
        self.__items.extend(items)

    def get(self) -> list[Item]:
        return list(self.__items)

    def get_new(self) -> Sequence[Item]:
        """Return only items added since the last cursor advance."""
        return self.__items[self.__live_cursor :]

    def advance_cursor(self) -> None:
        """Move cursor to the end of the current item list."""
        self.__live_cursor = len(self.__items)

    def clear(self) -> None:
        self.__items = []
        self.__live_cursor = 0
        self.current_tokens = 0
        self.usage = Usage()

    def load(self, items: Sequence[Item]) -> None:
        self.clear()
        for i in items:
            self.__items.append(i)


__all__ = ["History"]
