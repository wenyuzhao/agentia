from pathlib import Path

from pydantic import BaseModel
import pydantic
from . import Plugin, tool
from typing import Annotated
from datetime import datetime
from filelock import FileLock


class MemoryRecord(BaseModel):
    timestamp: datetime
    content: str


type MemoryRecords = list[MemoryRecord]


class MemoryPlugin(Plugin):
    def __init__(self, file: str | Path):
        super().__init__()
        file = Path(file)
        assert file.suffix == ".json", "Memory file must be a .json file"
        self.__memory_cache = Path(file)
        self.__records: MemoryRecords = []

    async def init(self):
        assert self.agent is not None, "Agent is required for MemoryPlugin"
        # Load existing memory records
        assert self.__memory_cache.exists()
        content = self.__memory_cache.read_text().strip()
        records = pydantic.TypeAdapter[MemoryRecords](MemoryRecords).validate_json(
            content
        )
        self.__records = records
        # Append to agent history
        self.agent.history.add_instructions(f"YOUR PREVIOUS MEMORY: \n{content}")

    @tool
    def remember(
        self,
        content: Annotated[str, "The content to remember. Keep it short and brief."],
    ):
        """Permanently remember something in your memory, as long as you think it's important or will be useful in the future. Use this to remember any important information whilst you are chatting with the user or fulfilling tasks."""
        with FileLock(str(self.__memory_cache) + ".lock"):
            record = MemoryRecord(timestamp=datetime.now(), content=content)
            self.__records.append(record)
            # save to file
            with open(self.__memory_cache) as f:
                f.write(
                    pydantic.TypeAdapter[MemoryRecords](MemoryRecords)
                    .dump_json(self.__records, indent=2)
                    .decode("utf-8")
                )
        return "Remembered"

    @tool
    def recall(self):
        """Recall all the things you remembered"""
        try:
            with open(self.__memory_cache, "r") as f:
                return f.read()
        except FileNotFoundError:
            return "I don't remember anything"
