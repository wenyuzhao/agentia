import dotenv
import pytest

from agentia import MCP, Agent

dotenv.load_dotenv()

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_mcp():
    agent = Agent(
        model="openai/gpt-5-nano",
        tools=[MCP(name="calculator", command="uvx", args=["mcp-server-calculator"])],
    )
    run = agent.run("Use tools to calculate 234 ** 3, don't add commas to the number")
    all_assistant_content: str = ""
    async for msg in run:
        if msg.type == "message" and msg.role == "assistant":
            all_assistant_content += msg.text
        print(msg)
    assert "12812904" in all_assistant_content
