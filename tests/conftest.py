"""Tests marked `live` call real model APIs: they only run with AGENTIA_LIVE_TESTS=1 (and
API keys). Everything else runs offline against mocked HTTP endpoints."""

import os

import pytest

LIVE = os.environ.get("AGENTIA_LIVE_TESTS") == "1"

if not LIVE:
    # Offline tests construct providers (which require a key) but never send it anywhere.
    os.environ.setdefault("OPENROUTER_API_KEY", "test-key")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if LIVE:
        return
    skip = pytest.mark.skip(reason="live API test; set AGENTIA_LIVE_TESTS=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
