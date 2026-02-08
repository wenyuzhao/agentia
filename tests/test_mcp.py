from agentia import Agent, MCP
import pytest
import dotenv


dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_mcp():
    agent = Agent(
        model="openai/gpt-5-nano",
        tools=[MCP(name="calculator", command="uvx", args=["mcp-server-calculator"])],
    )
    run = agent.run("Use tools to calculate 234 ** 3, don't add commas to the number")
    all_assistant_content: str = ""
    async for msg in run:
        if msg.role == "assistant":
            all_assistant_content += msg.text
        print(msg)
    assert "12812904" in all_assistant_content
