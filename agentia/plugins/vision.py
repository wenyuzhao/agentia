from typing import Annotated

from . import Plugin
from ..tools import tool
from openai import AsyncOpenAI


class VisionPlugin(Plugin):
    def __init__(self):
        super().__init__()
        self.client = AsyncOpenAI()

    @tool
    async def analyze_image(
        self,
        prompt: Annotated[
            str,
            "The prompt send to the vision model to analyze the image. Please ensure your prompt precisly and clearly state what do you want to know about the image, and provide more details or context in the prompt.",
        ],
        image_url: Annotated[str, "The URL of the image to analyze."],
    ):
        """Use gpt-4-vision-preview to analyze an image. Returning the analysis result."""
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Act as an image analyzer, and fulfill the users request. Please give your response in a verbose and detailed way to provide the user more information and context about the image. Here is the user's request:",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        )
        return response.choices[0].message.content or "No response from the model."
