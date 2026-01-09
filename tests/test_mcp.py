from agentia import Agent
import pytest
import dotenv
from agentia.tools.mcp import MCPServer, MCPContext


dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_mcp():
    async with MCPContext():
        agent = Agent(
            model="openai/gpt-5-nano",
            tools=[
                MCPServer(
                    name="calculator", command="uvx", args=["mcp-server-calculator"]
                )
            ],
        )
        run = agent.run("Calculate 234 ** 3, don't add commas to the number")
        all_assistant_content: str = ""
        async for msg in run:
            if msg.role == "assistant":
                all_assistant_content += msg.text
            print(msg)
        assert "12812904" in all_assistant_content
