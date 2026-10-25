"""Conversation history: an append-only list of items plus the compactions over it.

The history is the agent's state. A run never keeps anything important in memory: each
response item and each tool result is appended as soon as it exists, so a host that
persists `add()` can resume an interrupted run with `agent.run(None)`: pending tool
calls (calls without an output) are executed, otherwise the model is called again.

Compactions never rewrite the history. A `Compaction` summarizes the first `covers`
items; the model sees the latest summary followed by `context()`, the items after it.

To persist, subclass and override the storage methods: the `items` and `compactions`
properties, `_append()` and `_append_compaction()`.
"""

from collections.abc import Sequence

from agentia.models import (
    Compaction,
    FunctionCall,
    FunctionCallOutput,
    Item,
    Usage,
)


class History:
    def __init__(
        self,
        items: Sequence[Item] = (),
        compactions: Sequence[Compaction] = (),
    ) -> None:
        self._items: list[Item] = list(items)
        self._compactions: list[Compaction] = list(compactions)
        self.current_tokens: int = 0
        """Prompt + completion tokens of the latest model call (its context size)."""
        self.usage: Usage = Usage()
        """Accumulated usage of all runs."""

    # ----- storage (override to persist) -----

    @property
    def items(self) -> Sequence[Item]:
        """Every item ever added, in order. Compactions don't remove anything."""
        return self._items

    @property
    def compactions(self) -> Sequence[Compaction]:
        return self._compactions

    def _append(self, items: list[Item]) -> None:
        self._items.extend(items)

    def _append_compaction(self, compaction: Compaction) -> None:
        self._compactions.append(compaction)

    # ----- API -----

    def add(self, *items: Item) -> None:
        if items:
            self._append(list(items))

    def add_compaction(self, compaction: Compaction) -> None:
        last = self.compaction
        if last and compaction.covers < last.covers:
            raise ValueError(
                "a compaction cannot cover fewer items than the previous one"
            )
        if compaction.covers > len(self.items):
            raise ValueError("a compaction cannot cover items that don't exist")
        self._append_compaction(compaction)

    @property
    def compaction(self) -> Compaction | None:
        """The latest compaction, whose summary replaces the items it covers."""
        return self.compactions[-1] if self.compactions else None

    def context(self) -> list[Item]:
        """The items the model sees after the latest summary."""
        c = self.compaction
        return list(self.items[c.covers :] if c else self.items)

    def pending_tool_calls(self) -> list[FunctionCall]:
        """Function calls in the context that have no output yet, in call order."""
        items = self.context()
        answered = {i.call_id for i in items if isinstance(i, FunctionCallOutput)}
        return [
            i
            for i in items
            if isinstance(i, FunctionCall) and i.call_id not in answered
        ]

    def get(self) -> list[Item]:
        """A copy of all items."""
        return list(self.items)

    def clear(self) -> None:
        self._items = []
        self._compactions = []
        self.current_tokens = 0
        self.usage = Usage()

    def load(
        self, items: Sequence[Item], compactions: Sequence[Compaction] = ()
    ) -> None:
        self.clear()
        self._items = list(items)
        self._compactions = list(compactions)


def normalize_tool_pairs(items: Sequence[Item]) -> list[Item]:
    """Put every function call's output right after its group of calls.

    Outputs are appended to the history as tools finish, so they may be out of order or
    separated from their call (e.g. by a message added while a tool was suspended).
    Providers require each call to be followed by its output, so this moves outputs
    next to their calls, in call order, and synthesizes an error output for a call that
    has none. Outputs without a call are dropped.
    """
    outputs = {i.call_id: i for i in items if isinstance(i, FunctionCallOutput)}
    out: list[Item] = []
    group: list[FunctionCall] = []

    def flush() -> None:
        for c in group:
            o = outputs.get(c.call_id)
            out.append(
                o
                or FunctionCallOutput(
                    call_id=c.call_id,
                    name=c.name,
                    output="[no result recorded]",
                    is_error=True,
                )
            )
        group.clear()

    for item in items:
        if isinstance(item, FunctionCall):
            out.append(item)
            group.append(item)
            continue
        if group:
            flush()
        if isinstance(item, FunctionCallOutput):
            continue  # emitted right after its call (or dropped if it has none)
        out.append(item)
    flush()
    return out


__all__ = ["History", "normalize_tool_pairs"]
