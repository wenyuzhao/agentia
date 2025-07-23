from agentia import Agent
from agentia.message import UserMessage
from agentia.message import ContentPartImage, ContentPartText
import pytest
import dotenv

dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_vision():
    agent = Agent(model="gpt-4.1-nano")
    run = agent.run(
        UserMessage(
            content=[
                ContentPartText("What is this animal?"),
                ContentPartImage(
                    "https://icons.iconarchive.com/icons/iconarchive/cute-animal/256/Cute-Cat-icon.png"
                ),
            ],
        ),
    )
    all_assistant_content = ""
    async for msg in run:
        assert msg.content is None or isinstance(msg.content, str)
        all_assistant_content += msg.content or ""
        print(msg)
    assert "cat" in all_assistant_content.lower()
