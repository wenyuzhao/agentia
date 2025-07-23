from typing import Annotated, Union
from pydantic import BaseModel, Field

from .message import Message, SystemMessage, is_message, Event


class History(BaseModel):
    instructions: str | None = None
    messages: list[
        Annotated[Union[Message, "Event"], Field(union_mode="left_to_right")]
    ] = Field(default_factory=list)
    summary: str | None = None

    def __init__(self, instructions: str | None) -> None:
        super().__init__(
            instructions=instructions,
            messages=[],
            summary=None,
        )
        self.reset()

    def update_summary(self, summary: str) -> None:
        self.summary = summary

    def add_instructions(self, instructions: str) -> None:
        if self.instructions is None:
            self.instructions = instructions
        else:
            self.instructions += "\n\n" + instructions.strip()

    def get_for_inference(self) -> list[Message]:
        """
        Get the recent messages for inference
        """
        messages = History.__filter(self.messages)
        return messages

    def reset(self):
        self.messages = []
        if self.instructions is not None:
            self.add(SystemMessage(content=self.instructions))

    def add(self, message: Union[Message, "Event"]):
        # TODO: auto trim history
        self.messages.append(message)

    @staticmethod
    def __filter(
        msgs: list[Union[Message, "Event"]],
    ) -> list[Message]:
        """
        Filter out non-message events
        """
        return [m for m in msgs if is_message(m)]

    def get_formatted_history(self) -> str:
        """
        Get the formatted history for display
        """
        from .message import (
            UserMessage,
            AssistantMessage,
            ContentPartText,
            ContentPartImage,
        )

        S = ""
        for m in self.messages:
            match m:
                case UserMessage(content=content):
                    if isinstance(content, str):
                        S += f"@USER:\n{content}"
                    else:
                        S += "@USER:\n"
                        for p in content:
                            if isinstance(p, ContentPartText):
                                S += f"  - {p.content}\n"
                            elif isinstance(p, ContentPartImage):
                                S += f"  - {p.url}\n"
                case AssistantMessage(content=content, tool_calls=tool_calls):
                    S += f"@ASSISTANT:\n{content}\n"
                    if tool_calls:
                        S += "  - TOOL CALLS:\n"
                        for t in tool_calls:
                            S += f"    - [{t.id}] {t.function.name}\n"

        return S


__all__ = ["History"]
