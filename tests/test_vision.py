from agentia import Agent, UserMessage
from agentia.message import ContentPartImage, ContentPartText
import pytest
import dotenv

dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_vision():
    gpt = Agent(model="gpt-4o-mini")
    response = gpt.chat_completion(
        [
            UserMessage(
                content=[
                    ContentPartText("What is this animal?"),
                    ContentPartImage(
                        "https://icons.iconarchive.com/icons/iconarchive/cute-animal/256/Cute-Cat-icon.png"
                    ),
                ],
            ),
        ]
    )
    all_assistant_content = ""
    async for msg in response:
        assert msg.content is None or isinstance(msg.content, str)
        all_assistant_content += msg.content or ""
        print(msg)
    assert "cat" in all_assistant_content.lower()
