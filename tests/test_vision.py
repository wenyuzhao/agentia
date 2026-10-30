import io
import random

from PIL import Image, ImageDraw

from agentia import (
    Agent,
    AssistantMessage,
    File,
    ToolCallPart,
    ToolMessage,
    ToolResult,
    ToolResultPart,
)
from agentia.models import UserMessage, MessagePartText, MessagePartFile
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


def _square_png(color: str) -> bytes:
    image = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(image).rectangle((50, 50, 150, 150), fill=color)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model", ["anthropic/claude-haiku-4.5", "openai/gpt-5-mini"])
async def test_tool_generated_image_reaches_the_model(model: str, stream: bool):
    # Random, so that the model can only answer by seeing the image
    color = random.choice(["red", "green", "blue", "yellow"])

    def get_picture():
        """Returns the picture as an image attachment that you can look at."""
        return ToolResult(
            output="Picture attached.",
            files=[File(_square_png(color), media_type="image/png")],
        )

    agent = Agent(model=model, tools=[get_picture])
    prompt = "Call get_picture and tell me the color of the square in it."
    if stream:
        run = agent.run(prompt, stream=True)
        async for _ in run:
            pass
        response = await run
    else:
        response = await agent.run(prompt)
    assert any(isinstance(m, ToolMessage) for m in agent.history.get())
    assert color in response.text.lower()

    # The image stays in the history, so later turns can still see it
    response = await agent.run(
        "Look at the picture again: what is its background color? Don't call any tool."
    )
    assert "white" in response.text.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    ["anthropic/claude-haiku-4.5", "openai/gpt-5-mini", "google/gemini-2.5-flash"],
)
async def test_tool_image_in_history_is_sent_to_the_model(model: str):
    # Seeds the tool call, as whether a model decides to call a tool is not deterministic
    color = random.choice(["red", "green", "blue", "yellow"])

    def get_picture():
        """Returns the picture as an image attachment that you can look at."""
        return ToolResult(output="Picture attached.")

    call = ToolCallPart(tool_call_id="call_1", tool_name="get_picture", input={})
    result = ToolResultPart(
        tool_call_id="call_1",
        tool_name="get_picture",
        input={},
        output="Picture attached.",
        output_files=[File(_square_png(color), media_type="image/png")],
    )
    agent = Agent(model=model, tools=[get_picture])
    agent.history.load(
        [
            UserMessage("Get the picture."),
            AssistantMessage([call]),
            ToolMessage([result]),
        ]
    )
    response = await agent.run(
        "What is the color of the square in the picture? Don't call any tool."
    )
    assert color in response.text.lower()
