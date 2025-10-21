import pytest
import dotenv
from agentia.tools.mcp import MCPContext
from agentia.utils.config import load_agent_from_config

dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_load_agent_from_config():
    agent = load_agent_from_config("./examples/example-agent.toml")
    async with MCPContext():
        run = agent.run("Calculate 2 ^ 12")
        response = await run
        result_text = response.text
    assert "4096" in result_text
