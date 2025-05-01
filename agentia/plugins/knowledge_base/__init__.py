from io import BytesIO
from pathlib import Path
from typing import Annotated, Union

from agentia.plugins.knowledge_base.knowledge_base import KnowledgeBase
from .. import Plugin
from agentia.tools import tool


TOP_K = 4


class KnowledgeBasePlugin(Plugin):
    def __init__(self, source: Union["KnowledgeBase", Path | str, None] = None):
        self.source = source
        self.__pending_files: list[BytesIO | Path | str] = []
        self.knowledge_base: KnowledgeBase | None = None

    async def init(self):
        # Load knowledge base
        if isinstance(self.source, KnowledgeBase):
            # Load a pre-existing knowledge base
            self.knowledge_base = self.source
        elif isinstance(self.source, Path | str):
            # Load knowledge base from a collection of documents
            self.knowledge_base = KnowledgeBase(global_docs=Path(self.source))
        else:
            # Create a new knowledge base without global documents
            self.knowledge_base = KnowledgeBase()
        # Add session store
        self.knowledge_base._create_session_store(self.agent)
        # Add pending files
        for f in self.__pending_files:
            if isinstance(f, str):
                if f.startswith("http://") or f.startswith("https://"):
                    self.knowledge_base.add_doc_from_url(f)
                    continue
                else:
                    f = Path(f)
            self.knowledge_base.add_document(f)
        self.__pending_files = []
        # Update instructions
        files = []
        if vs := self.knowledge_base.vector_stores.get("global"):
            files += vs.get_indexed_files()
        if vs := self.knowledge_base.vector_stores.get("session"):
            files += vs.get_indexed_files()
        # print("FILES IN KNOWLEDGE-BASE", files)
        if len(files) > 0:
            self.agent.history.add_instructions(
                f"KNOWLEDGE-BASE ENABLED! Please use the file search tool to search for files in the knowledge base.\n"
                f"FILES IN KNOWLEDGE-BASE: {', '.join(files)}"
            )

    def add_file(self, file: Path | str | BytesIO):
        """Add a file to the knowledge base"""
        if self.knowledge_base is None:
            self.__pending_files.append(file)
        else:
            if isinstance(file, str):
                if file.startswith("http://") or file.startswith("https://"):
                    self.knowledge_base.add_doc_from_url(file)
                    return
                else:
                    file = Path(file)
            self.knowledge_base.add_document(file)

    @tool
    async def file_search(
        self, query: Annotated[str, "The query to search for in the knowledge base"]
    ):
        """Search for related file segments in the knowledge base"""
        print(f"FILE-SEARCH {query}")
        self.agent.log.debug(f"FILE-SEARCH {query}")
        assert self.knowledge_base is not None
        items = await self.knowledge_base.query(query)
        if len(items) == 0:
            return "No results found"
        return [x.to_dict() for x in items]

    @tool
    async def _add_file(self, url: Annotated[str, "The URL of the file to retrieve"]):
        """Download a file from the internet and add it to the knowledge base"""
        print(f"EMBED {url}")
        self.agent.log.debug(f"EMBED {url}")
        assert self.knowledge_base is not None
        if r := self.knowledge_base.add_doc_from_url(url):
            return r
        raise ValueError(
            f"Failed to add file from {url}: Unsupported file type or error occurred."
        )
