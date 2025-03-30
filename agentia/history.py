from typing import TYPE_CHECKING, Any, Union

import tiktoken
from .message import BaseMessage, Message, SystemMessage

ENCODING = tiktoken.encoding_for_model("gpt-4o-mini")

if TYPE_CHECKING:
    from .agent import Event


class History:
    def __init__(self, instructions: str | None) -> None:
        self._instructions = instructions
        self.__messages: list[Union[Message, "Event"]] = []
        self.reset()

    def get_for_inference(self, keep_last=0) -> list[Message]:
        """
        Get the recent messages for inference
        """
        messages = History.__trim_and_filter(self.__messages, keep_last=keep_last)
        return messages

    def reset(self):
        self.__messages = []
        if self._instructions is not None:
            self.add(SystemMessage(self._instructions))

    def add(self, message: Union[Message, "Event"]):
        # TODO: auto trim history
        self.__messages.append(message)

    def set(self, messages: list[Union[Message, "Event"]]):
        self.__messages = messages

    def get(self) -> list[Union[Message, "Event"]]:
        return self.__messages

    @staticmethod
    def __trim_and_filter(
        msgs: list[Union[Message, "Event"]],
        token_limit: int = 120000,
        keep_first=0,
        keep_last=0,
    ) -> list[Message]:
        from .message import ContentPartText

        first_system_message = None
        if len(msgs) > 0 and isinstance(msgs[0], SystemMessage):
            first_system_message = msgs[0]
            other_messages = msgs[1:]
            if keep_first > 0:
                keep_first -= 1
        else:
            other_messages = msgs
        head_msgs = []
        if keep_first > 0:
            head_msgs = other_messages[:keep_first]
            other_messages = other_messages[keep_first:]
        tail_msgs = []
        if keep_last > 0:
            tail_msgs = other_messages[-keep_last:]
            other_messages = other_messages[:-keep_last]

        def count_tokens(m: Message) -> int:
            if isinstance(m.content, str):
                return len(ENCODING.encode(m.content))
            elif isinstance(m.content, list):
                tokens = 0
                for part in m.content:
                    if isinstance(part, ContentPartText):
                        tokens += len(ENCODING.encode(part.content))
                return tokens
            return 0

        msgs2: list[Message] = []
        tokens = 0
        if first_system_message is not None:
            msgs2.append(first_system_message)
            tokens += count_tokens(first_system_message)
        for m in head_msgs:
            if not isinstance(m, Message):
                continue
            msgs2.append(m)
            tokens += count_tokens(m)
        for m in tail_msgs:
            if not isinstance(m, Message):
                continue
            tokens += count_tokens(m)
        for m in other_messages:
            if not isinstance(m, Message):
                continue
            tokens += count_tokens(m)
            if tokens > token_limit:
                break
            msgs2.append(m)
        msgs2.extend([m for m in tail_msgs if isinstance(m, Message)])
        return msgs2
