import inspect
import shlex
from typing import TYPE_CHECKING, Awaitable, Callable, Sequence

from agentia.models.chat import Message, MessagePartText, UserMessage

if TYPE_CHECKING:
    from agentia.agent import Agent


CommandHandler = Callable[[list[str]], str | None | Awaitable[str | None]]


class Commands:
    """
    Slash command dispatcher for an `Agent`.

    Parses inputs of the form `/name arg0 arg1 ...` and dispatches to:
        1. A custom handler registered via `register(name, handler)`.
        2. Otherwise, a user-invocable skill with the same name (if the
           agent has the `Skills` plugin loaded).
        3. Otherwise, does nothing.
    """

    def __init__(self, agent: "Agent") -> None:
        self.__agent = agent
        self.__handlers: dict[str, CommandHandler] = {}

    def register(self, name: str, handler: CommandHandler) -> None:
        """Register a custom slash command handler. The leading `/` is optional."""
        self.__handlers[_strip_slash(name)] = handler

    def unregister(self, name: str) -> None:
        """Remove a previously registered custom handler. No-op if absent."""
        self.__handlers.pop(_strip_slash(name), None)

    def has(self, name: str) -> bool:
        """Return True if `name` resolves to a custom handler or a user-invocable skill."""
        n = _strip_slash(name)
        if n in self.__handlers:
            return True
        return self.__find_skill(n) is not None

    @staticmethod
    def parse(input: str) -> tuple[str, list[str]] | None:
        """
        Parse `/name arg0 arg1 ...` into `(name, [arg0, arg1, ...])`.
        Returns `None` if `input` is not a slash command or is malformed.
        """
        if not input.startswith("/"):
            return None
        try:
            tokens = shlex.split(input[1:])
        except ValueError:
            return None
        if not tokens:
            return None
        return tokens[0], tokens[1:]

    async def handle(self, input: str) -> str | None:
        """
        Dispatch a slash command. Returns the handler/skill output, or `None`
        if the input is not a slash command or no handler/skill matched.
        """
        parsed = self.parse(input)
        if parsed is None:
            return input
        name, args = parsed
        if handler := self.__handlers.get(name):
            result = handler(args)
            if inspect.isawaitable(result):
                result = await result
            if result and isinstance(result, str) and not result:
                result = None
            return result
        if skill := self.__find_skill(name):
            return skill.execute(args)
        return input  # treat as normal input if no command matched

    async def process_messages(
        self, prompt: str | Message | Sequence[Message]
    ) -> list[Message]:
        messages = []
        for p in (
            prompt
            if isinstance(prompt, Sequence) and not isinstance(prompt, str)
            else [prompt]
        ):
            if isinstance(p, str):
                if p := await self.handle(p):
                    messages.append(UserMessage(p))
            elif isinstance(p, UserMessage):
                if isinstance(p.content, str):
                    if c := await self.handle(p.content):
                        messages.append(UserMessage(c))
                else:
                    parts = []
                    for part in p.content:
                        if isinstance(part, MessagePartText):
                            if c := await self.handle(part.text):
                                parts.append(MessagePartText(c))
                        else:
                            parts.append(part)
                    if parts:
                        messages.append(UserMessage(parts))
            else:
                messages.append(p)
        return messages

    def __find_skill(self, name: str):
        from agentia.plugins.skills import Skill, Skills

        skills = self.__agent.tools.get_plugin(Skills)
        if skills is None:
            return None
        skill: Skill | None = skills.user_invocable_skills.get(name)
        return skill


def _strip_slash(name: str) -> str:
    return name[1:] if name.startswith("/") else name
