from agentia.spec import *


class History:
    def __init__(self) -> None:
        self.__instructions: list[str] = []
        self.__non_instruction_messages: list[
            UserMessage | AssistantMessage | ToolMessage
        ] = []

    def add_instructions(self, instruction: str) -> None:
        self.__instructions.append(instruction)

    def __get_instructions(self) -> str:
        return "\n\n".join(self.__instructions)

    def add(self, *messages: UserMessage | AssistantMessage | ToolMessage) -> None:
        self.__non_instruction_messages.extend(messages)

    def get(self, include_instructions: bool = True) -> list[Message]:
        messages: list[Message] = []
        if include_instructions:
            if i := self.__get_instructions():
                messages.append(SystemMessage(content=i))
        messages.extend(self.__non_instruction_messages)
        return messages

    def clear(self) -> None:
        self.__non_instruction_messages = []

    def load(
        self, messages: Sequence[Message], exclude_instructions: bool = False
    ) -> None:
        self.clear()
        for m in messages:
            if isinstance(m, SystemMessage):
                if not exclude_instructions:
                    self.__instructions.append(m.content)
            else:
                self.__non_instruction_messages.append(m)


__all__ = ["History"]
