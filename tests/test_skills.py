from agentia import Agent
import pytest
import dotenv

dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_skills():
    agent = Agent(model="openai/gpt-5-nano", skills=["./examples"])
    run = agent.run("What is the weather and temperature like in boston?")
    all_assistant_content: str = ""
    async for msg in run:
        if msg.role == "assistant":
            all_assistant_content += msg.text
        print(msg)
    assert "27" in all_assistant_content
