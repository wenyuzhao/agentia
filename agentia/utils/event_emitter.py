import inspect
from typing import Callable, overload


class EventEmitter[EventHandler: Callable]:
    def __init__(self):
        self._listeners: set[EventHandler] = set()

    @overload
    def on(self, listener: EventHandler): ...
    @overload
    def on(self) -> Callable[[EventHandler], EventHandler]: ...

    def on(self, listener: EventHandler | None = None):
        if listener is None:

            def decorator(func: EventHandler, /):
                self.on(func)
                return func

            return decorator
        self._listeners.add(listener)

    @overload
    def off(self, listener: EventHandler): ...
    @overload
    def off(self) -> Callable[[EventHandler], EventHandler]: ...

    def off(self, listener: EventHandler | None = None):
        if listener is None:

            def decorator(func: EventHandler, /):
                self.off(func)
                return func

            return decorator
        self._listeners.remove(listener)

    def __len__(self):
        return len(self._listeners)

    async def emit(self, *args, **kwargs):
        results = []
        for listener in self._listeners:
            r = listener(*args, **kwargs)
            if inspect.isawaitable(r):
                r = await r
            results.append(r)
        return results
