from agentia import Agent
from agentia.spec import UserMessage, MessagePartText, MessagePartFile
import pytest
import dotenv

dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_vision():
    agent = Agent(model="gpt-5-nano")
    run = agent.run(
        UserMessage(
            content=[
                MessagePartText(text="What is this animal?"),
                MessagePartFile(data="https://cataas.com/cat", media_type="image/png"),
            ],
        ),
    )
    all_assistant_content = ""
    async for msg in run:
        if msg.role == "assistant":
            all_assistant_content += msg.text
        print(msg)
    assert "cat" in all_assistant_content.lower()
