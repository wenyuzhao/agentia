import pytest
import dotenv

dotenv.load_dotenv()

from agentia import Agent  # noqa: E402
from agentia.llm import get_provider  # noqa: E402


class TestModelImmutability:
    def test_agent_model_is_readonly(self):
        agent = Agent(model="openai/gpt-5-nano")
        assert agent.model == "openai/gpt-5-nano"
        with pytest.raises(AttributeError):
            agent.model = "openai/gpt-5-mini"  # type: ignore[misc]

    def test_provider_model_is_readonly(self):
        provider = get_provider("openai:gpt-5-nano")
        assert provider.model == "gpt-5-nano"
        with pytest.raises(AttributeError):
            provider.model = "gpt-5-mini"  # type: ignore[misc]

    def test_provider_model_strips_think_suffix(self):
        provider = get_provider("openai:gpt-5-nano:think")
        assert provider.model == "gpt-5-nano"

    def test_openrouter_model_is_readonly(self):
        provider = get_provider("openrouter:openai/gpt-5-nano")
        assert provider.model == "openai/gpt-5-nano"
        with pytest.raises(AttributeError):
            provider.model = "openai/gpt-5-mini"  # type: ignore[misc]


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
        provider = get_provider("openai:gpt-5-nano")
        ctx = await provider.get_context_length()
        assert isinstance(ctx, int)
        assert ctx > 0

    @pytest.mark.asyncio
    async def test_agent_context_length(self):
        agent = Agent(model="openai/gpt-5-nano")
        ctx = await agent.get_max_context_length()
        assert isinstance(ctx, int)
        assert ctx > 0
