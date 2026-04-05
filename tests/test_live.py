import asyncio
import pytest
import dotenv
from agentia import Agent, LiveOptions

dotenv.load_dotenv()

LIVE_MODEL = "gemini-live:gemini-3.1-flash-live-preview"


@pytest.mark.asyncio
async def test_live_text():
    """Test connecting, sending text, and receiving audio + transcription response."""
    agent = Agent(
        model=LIVE_MODEL,
        instructions="You are a helpful assistant. Keep your responses very short.",
        live_options=LiveOptions(
            modalities=["audio"],
            enable_output_transcription=True,
        ),
    )
    received_transcription = ""
    received_audio = False
    async with agent:
        await agent.send_text("What is 2 + 2? Reply with just the number.")
        async for event in agent.receive():
            if event.type == "audio-delta":
                received_audio = True
            elif event.type == "output-transcription-delta":
                received_transcription += event.delta
            elif event.type == "turn-end":
                break
    assert received_audio or received_transcription
    if received_transcription:
        assert "4" in received_transcription


@pytest.mark.asyncio
async def test_live_tool_calling():
    """Test tool calling with auto execution in a live session."""
    tool_called = False

    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        nonlocal tool_called
        tool_called = True
        return f"Sunny, 72°F in {location}"

    agent = Agent(
        model=LIVE_MODEL,
        instructions="You are a helpful assistant. Use tools when needed. Keep responses short.",
        tools=[get_weather],
        live_options=LiveOptions(
            modalities=["audio"],
            enable_output_transcription=True,
        ),
    )
    received_transcription = ""
    async with agent:
        await agent.send_text("What is the weather in Boston?")
        async for event in agent.receive():
            if event.type == "output-transcription-delta":
                received_transcription += event.delta
            elif event.type == "tool-call":
                assert event.tool_name == "get_weather"
            elif event.type == "tool-result":
                assert "Sunny" in str(event.output)
            elif event.type == "turn-end":
                break
    assert tool_called


@pytest.mark.asyncio
async def test_live_run_emulation():
    """Test that agent.run() works over a live session (request/response emulation)."""
    agent = Agent(
        model=LIVE_MODEL,
        instructions="You are a helpful assistant. Keep responses short.",
        live_options=LiveOptions(modalities=["audio"]),
    )
    async with agent:
        result = await agent.run("What is 3 + 5? Reply with just the number.")
        # With audio modality, run() may get empty text since output is audio
        # Just verify it doesn't crash and returns a result
        assert result is not None


@pytest.mark.asyncio
async def test_live_parallel_send_receive():
    """Test concurrent send and receive in parallel async tasks."""
    agent = Agent(
        model=LIVE_MODEL,
        instructions="You are a helpful assistant. Keep your responses very short. Reply to each question independently.",
        live_options=LiveOptions(
            modalities=["audio"],
            enable_output_transcription=True,
        ),
    )
    questions = [
        "What is 10 + 20? Reply with just the number.",
        "What is the capital of France? Reply with just the city name.",
    ]
    received_transcription = ""
    received_audio = False
    turns_completed = 0
    turn_complete_event = asyncio.Event()

    async with agent:

        async def sender():
            for i, q in enumerate(questions):
                if i > 0:
                    # Wait for previous turn to complete before sending next
                    await turn_complete_event.wait()
                    turn_complete_event.clear()
                await agent.send_text(q)

        async def receiver():
            nonlocal received_transcription, received_audio, turns_completed
            async for event in agent.receive():
                if event.type == "audio-delta":
                    received_audio = True
                elif event.type == "output-transcription-delta":
                    received_transcription += event.delta
                elif event.type == "turn-end":
                    turns_completed += 1
                    turn_complete_event.set()
                    if turns_completed >= len(questions):
                        break

        send_task = asyncio.create_task(sender())
        recv_task = asyncio.create_task(receiver())
        await asyncio.gather(send_task, recv_task)

    assert turns_completed >= len(questions)
    assert received_audio or received_transcription
