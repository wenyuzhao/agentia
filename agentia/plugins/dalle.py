from typing import Annotated

from . import Plugin
from ..tools import tool
from openai import AsyncOpenAI


class DallEPlugin(Plugin):
    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.client = AsyncOpenAI(api_key=api_key)

    @tool
    async def generate_image(
        self,
        prompt: Annotated[
            str,
            "The prompt to generate an image from. Please keep your prompt verbose and precise, and provide more details or context in the prompt.",
        ],
    ):
        """Use Dall-E 3 to generate an image from a prompt. Returning the generated image."""
        response = await self.client.images.generate(
            prompt=prompt,
            model="dall-e-3",
        )
        if not response.data or len(response.data) == 0:
            raise ValueError("No image generated.")
        return {
            "image_url": response.data[0].url,
        }
