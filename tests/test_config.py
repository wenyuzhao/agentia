from agentia import Agent
import pytest
import dotenv

dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_load_agent_from_config():
    agent = Agent.load_from_config("./examples/example-agent.toml")
    run = agent.run("Calculate 2 ^ 12")
    response = await run
    assert "4096" in (response.content or "")
