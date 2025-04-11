from typing import (
    Callable,
    Coroutine,
    Optional,
    TypeVar,
    Any,
    overload,
)

from agentia.message import JSON


R = TypeVar("R", Coroutine[Any, Any, Optional[JSON | str]], Optional[JSON | str])


@overload
def tool(name: Callable[..., R]) -> Callable[..., R]: ...


@overload
def tool(
    name: str | None = None,
    display_name: str | None = None,
    description: str | None = None,
) -> Callable[..., Callable[..., R]]: ...


def tool(
    name: str | Callable[..., R] | None = None,
    display_name: str | None = None,
    description: str | None = None,
) -> Callable[..., R] | Callable[[Callable[..., R]], Callable[..., R]]:

    def __tool_impl(callable: Callable[..., R]) -> Callable[..., R]:
        # store gpt function metadata to the callable object
        from agentia.tools import (
            IS_TOOL_TAG,
            NAME_TAG,
            DISPLAY_NAME_TAG,
            DESCRIPTION_TAG,
        )

        if isinstance(name, str):
            setattr(callable, NAME_TAG, name)

        if isinstance(display_name, str):
            setattr(callable, DISPLAY_NAME_TAG, display_name)

        if isinstance(description, str):
            setattr(callable, DESCRIPTION_TAG, description)

        setattr(callable, IS_TOOL_TAG, True)

        return callable

    if (
        name is not None
        and (not isinstance(name, str))
        and display_name is None
        and description is None
    ):
        return __tool_impl(name)

    return __tool_impl


__all__ = ["tool"]
