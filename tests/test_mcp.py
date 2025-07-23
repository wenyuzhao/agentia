from agentia import Agent
import pytest
import dotenv
from agentia.mcp import LocalMCPServer, MCPContext


dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_mcp():
    async with MCPContext():
        agent = Agent(
            model="openai/gpt-4o-mini",
            tools=[
                LocalMCPServer(
                    name="calculator",
                    cmd=["uvx", "mcp-server-calculator"],
                )
            ],
        )
        run = agent.run("Calculate 234 ** 3, don't add commas to the number")
        all_assistant_content: str = ""
        async for msg in run:
            if msg.role == "assistant":
                assert msg.content is None or isinstance(msg.content, str)
                all_assistant_content += msg.content or ""
            print(msg)
        assert "12812904" in all_assistant_content
