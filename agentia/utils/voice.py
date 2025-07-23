from typing import Literal

import openai
from pathlib import Path
import os

Voice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


async def tts(
    text: str,
    output: str | Path,
    model: Literal["tts-1", "tts-1-hd"] = "tts-1",
    voice: Voice = "alloy",
    api_key: str | None = None,
):
    client = openai.AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    response = client.audio.speech.with_streaming_response.create(
        model=model, voice=voice, input=text
    )
    async with response as r:
        await r.stream_to_file(output)


async def stt(
    path: str | Path,
    model: Literal["whisper-1"] = "whisper-1",
    api_key: str | None = None,
) -> str:
    client = openai.AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    with open(path, "rb") as audio_file:
        transcript = await client.audio.transcriptions.create(
            model=model, file=audio_file
        )
        return transcript.text
