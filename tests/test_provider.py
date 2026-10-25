import os

import dotenv
import pytest

dotenv.load_dotenv()

from agentia import Agent
from agentia.llm import get_provider
from agentia.llm.providers.openai import OpenAI


class TestModelImmutability:
    def test_agent_model_is_readonly(self):
        agent = Agent(model="openai/gpt-5-nano")
        assert agent.model == "openai/gpt-5-nano"
        with pytest.raises(AttributeError):
            agent.model = "openai/gpt-5-mini"  # type: ignore[misc]

    def test_provider_model_is_readonly(self):
        # No provider prefix, so this goes through the default provider.
        provider = get_provider("openai/gpt-5-nano")
        assert provider.model == "openai/gpt-5-nano"
        with pytest.raises(AttributeError):
            provider.model = "openai/gpt-5-mini"  # type: ignore[misc]

    def test_provider_model_strips_think_suffix(self):
        provider = get_provider("openrouter:openai/gpt-5-nano:think")
        assert provider.model == "openai/gpt-5-nano"

    def test_openrouter_model_is_readonly(self):
        provider = get_provider("openrouter:openai/gpt-5-nano")
        assert provider.model == "openai/gpt-5-nano"
        with pytest.raises(AttributeError):
            provider.model = "openai/gpt-5-mini"  # type: ignore[misc]


@pytest.mark.live
class TestGetContextLength:
    @pytest.mark.asyncio
    async def test_openrouter_context_length(self):
        provider = get_provider("openrouter:openai/gpt-5-nano")
        ctx = await provider.get_context_length()
        assert isinstance(ctx, int)
        assert ctx > 0

    @pytest.mark.asyncio
    async def test_openrouter_context_length_cached(self):
        provider = get_provider("openrouter:openai/gpt-5-nano")
        ctx1 = await provider.get_context_length()
        ctx2 = await provider.get_context_length()
        assert ctx1 == ctx2

    @pytest.mark.asyncio
    async def test_openai_context_length(self):
        # The OpenAI provider has no context length in its own API and falls back to
        # OpenRouter's public model catalog, so the OpenRouter key is enough here.
        provider = OpenAI(model="gpt-5-nano", api_key=os.environ["OPENROUTER_API_KEY"])
        ctx = await provider.get_context_length()
        assert isinstance(ctx, int)
        assert ctx > 0

    @pytest.mark.asyncio
    async def test_agent_context_length(self):
        agent = Agent(model="openai/gpt-5-nano")
        ctx = await agent.get_max_context_length()
        assert isinstance(ctx, int)
        assert ctx > 0
