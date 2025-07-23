from pathlib import Path
from typing import TYPE_CHECKING, Union
import shelve

from .message import Message, SystemMessage, is_message

if TYPE_CHECKING:
    from .agent import Event


class History:
    def __init__(self, instructions: str | None) -> None:
        self._instructions = instructions
        self.__messages: list[Union[Message, "Event"]] = []
        self.summary = None
        self.__first_conversation_finished = False
        self.reset()

    def update_summary(self, summary: str) -> None:
        self.summary = summary

    def add_instructions(self, instructions: str) -> None:
        if self._instructions is None:
            self._instructions = instructions
        else:
            self._instructions += "\n\n" + instructions.strip()

    def get_for_inference(self) -> list[Message]:
        """
        Get the recent messages for inference
        """
        messages = History.__filter(self.__messages)
        return messages

    def reset(self):
        self.__messages = []
        if self._instructions is not None:
            self.add(SystemMessage(content=self._instructions))

    def add(self, message: Union[Message, "Event"]):
        # TODO: auto trim history
        self.__messages.append(message)

    def set(self, messages: list[Union[Message, "Event"]]):
        self.__messages = messages

    def get(self) -> list[Union[Message, "Event"]]:
        return self.__messages

    @staticmethod
    def __filter(
        msgs: list[Union[Message, "Event"]],
    ) -> list[Message]:
        """
        Filter out non-message events
        """
        return [m for m in msgs if is_message(m)]

    def _first_conversation_just_finished(self) -> bool:
        from .message import AssistantMessage, UserMessage

        if self.__first_conversation_finished:
            return False
        has_assistant = (
            len([m for m in self.__messages if isinstance(m, AssistantMessage)]) > 0
        )
        has_user = len([m for m in self.__messages if isinstance(m, UserMessage)]) > 0
        finished = has_assistant and has_user
        self.__first_conversation_finished = finished
        return finished

    def _save(self, path: Path) -> None:
        """
        Save the history to a file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        from .message import AssistantMessage

        with shelve.open(path) as db:
            db["messages"] = self.__messages
            if self.summary:
                db["summary"] = self.summary

    def _load(self, path: Path) -> None:
        """
        Load the history from a file
        """
        if not path.exists():
            return
        with shelve.open(path) as db:
            if "messages" in db:
                self.__messages = db["messages"]
            if "messages" in db:
                self.summary = db.get("summary", None)

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
        for m in self.__messages:
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
