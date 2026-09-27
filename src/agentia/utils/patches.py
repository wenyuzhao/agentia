from typing import Any, AsyncGenerator, Generator
from agentia.llm.completion import async_gen_to_sync


def patch_asyncio():
    import nest_asyncio

    nest_asyncio.apply()


def patch_streamlit():
    import streamlit
    from streamlit import type_util

    if not hasattr(streamlit, "_async_generator_to_sync_patched"):

        def async_generator_to_sync(
            async_gen: AsyncGenerator[Any, Any],
        ) -> Generator[Any, Any, Any]:
            return async_gen_to_sync(async_gen)

        type_util.async_generator_to_sync = async_generator_to_sync
        setattr(streamlit, "_async_generator_to_sync_patched", True)


def patch_all():
    def try_patch(func):
        try:
            func()
        except ImportError:
            ...

    try_patch(patch_asyncio)
    try_patch(patch_streamlit)
