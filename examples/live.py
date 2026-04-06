import asyncio

from agentia import Agent, load_dotenv
from agentia.live import InputStream, LiveChunkText, LiveOptions


class TextInput(InputStream):
    async def read(self) -> str:
        return await asyncio.to_thread(input, "> ")

    async def run(self):
        while True:
            text = await self.read()
            if not text:
                continue
            if text.lower() in ("exit", "quit"):
                raise KeyboardInterrupt()
            await self.send(LiveChunkText(text=text))


async def main():
    agent = Agent("gemini-live:gemini-3.1-flash-live-preview")
    await agent.live(options=LiveOptions(voice="Zephyr")).start(
        inputs=["audio", "screen"], outputs=["audio"]
    )


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
