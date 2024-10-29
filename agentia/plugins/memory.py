from ..decorators import *
from . import Plugin
from typing import Annotated
from datetime import datetime
from filelock import FileLock


class MemoryPlugin(Plugin):
    async def init(self):
        self.__momery_cache = self.agent.agent_data_folder / "memory"

    @tool
    def remember(
        self,
        content: Annotated[str, "The content to remember. Keep it short and brief."],
    ):
        """Permanently remember something in your memory, as long as you think it's important or will be useful in the future. Use this to remember any important information whilst you are chatting with the user or fulfilling tasks."""
        with FileLock(str(self.__momery_cache) + ".lock"):
            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.__momery_cache, "a") as f:
                f.write(f"[{time}] {content}\n")
            self.log.info(f"REMEMBER: {content}")
        return "Remembered"

    @tool
    def recall(self):
        """Recall all the things you remembered"""
        try:
            with open(self.__momery_cache, "r") as f:
                return f.read()
        except FileNotFoundError:
            return "I don't remember anything"
