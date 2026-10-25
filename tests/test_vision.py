import dotenv
import pytest

from agentia import Agent
from agentia.models import File, Message

dotenv.load_dotenv()

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_vision():
    agent = Agent(model="gpt-5-nano")
    run = agent.run(
        Message.user(
            [
                "What is this animal?",
                File(data="https://cataas.com/cat", media_type="image/png"),
            ]
        ),
    )
    all_assistant_content = ""
    async for msg in run:
        if msg.type == "message" and msg.role == "assistant":
            all_assistant_content += msg.text
        print(msg)
    assert "cat" in all_assistant_content.lower()
