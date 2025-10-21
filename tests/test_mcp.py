from agentia import Agent
import pytest
import dotenv
from agentia.mcp import MCPServer, MCPContext


dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_mcp():
    async with MCPContext():
        agent = Agent(
            model="openai/gpt-4.1-nano",
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
                for p in msg.content:
                    if p.type == "text":
                        all_assistant_content += p.text or ""
            print(msg)
        assert "12812904" in all_assistant_content
