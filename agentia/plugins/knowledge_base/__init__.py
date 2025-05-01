from pathlib import Path
from typing import Annotated, Any, Union

from agentia.knowledge_base import KnowledgeBase
from .. import Plugin, tool


class KnowledgeBasePlugin(Plugin):
    def __init__(self, source: Union["KnowledgeBase", Path, None] = None):
        self.source = source

    async def init(self):
        # Get session store persist path
        session_store = self.agent.session_data_folder / "knowledge-base"
        session_store.mkdir(parents=True, exist_ok=True)
        # Load knowledge base
        if isinstance(self.source, KnowledgeBase):
            # Load a pre-existing knowledge base
            knowledge_base = self.source
            knowledge_base.add_session_store(session_store)
        elif isinstance(self.source, Path):
            # Load knowledge base from a collection of documents
            persist_dir = self.agent.agent_data_folder / "knowledge-base"
            knowledge_base = KnowledgeBase(
                global_store=persist_dir,
                global_docs=self.source,
                session_store=session_store,
            )
        else:
            # Create a new knowledge base or load a pre-existing one
            knowledge_base = KnowledgeBase(
                global_store=self.agent.agent_data_folder / "knowledge-base",
                session_store=session_store,
            )
        # Update instructions
        global_vector_store = knowledge_base.vector_stores["global"]
        if len(global_vector_store.initial_files or []) > 0:
            files = global_vector_store.initial_files or []
            self.agent.history.add_instructions(
                f"FILES IN KNOWLEDGE-BASE: {', '.join(files)}"
            )

    @tool
    async def file_search(
        self,
        query: Annotated[str, "The query to search for files"],
        filename: Annotated[
            str | None,
            "The optional filename of the file to search from. If not provided, search from all files.",
        ] = None,
    ):
        """Similarity-based search for related file segments in the knowledge base"""
        self.agent.log.debug(f"FILE-SEARCH {query} ({filename or "<all-files>"})")
        assert self.agent.knowledge_base is not None
        response = await self.agent.knowledge_base.query(query, filename)
        return response
