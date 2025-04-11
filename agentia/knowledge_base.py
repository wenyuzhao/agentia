from io import BytesIO
from pathlib import Path
from llama_index.core.query_engine import CitationQueryEngine
import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import NodeWithScore
from agentia.utils.retrieval.vector_store import VectorStore, is_file_supported
from agentia.utils.retrieval.retriever import TOP_K, MultiRetriever


class KnowledgeBase:
    def __init__(
        self,
        global_store: Path,
        global_docs: Path | None = None,
        session_store: Path | None = None,
    ):
        """
        Create or load a knowledge base.
        It will load vector stores from both a global store and a session store (if provided).
        """

        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY must be set to enable the knowledge base")

        persist_dir = global_store
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.vector_stores = {
            "global": VectorStore(
                persist_path=persist_dir,
                docs=global_docs or persist_dir / "docs",
            ),
        }
        if session_store is not None:
            self.vector_stores["session"] = VectorStore(persist_path=session_store)
        self.__persist_dir = persist_dir
        self.__retriever = MultiRetriever(
            vector_stores=list(self.vector_stores.values()), file=None
        )
        self.__query_engine = CitationQueryEngine.from_args(
            self.vector_stores["global"].index,
            similarity_top_k=TOP_K,
            citation_chunk_size=1024,
            retriever=self.__retriever,
        )

    def add_session_store(self, session_store: Path):
        self.vector_stores["session"] = VectorStore(persist_path=session_store)
        self.__retriever.vector_stores = list(self.vector_stores.values())

    @staticmethod
    def is_file_supported(file_ext: str) -> bool:
        return is_file_supported(file_ext)

    async def query(self, query: str, file: str | None) -> str:
        """Query the knowledge base"""
        self.__retriever.file = file
        response = await self.__query_engine.aquery(query)
        if len(response.source_nodes) == 0:
            return "ERROR: No results found because the knowledge base is empty."
        formatted_response = str(response) + "\n\n\nSOURCES:\n\n"
        for i, node in enumerate(response.source_nodes):
            assert isinstance(node, NodeWithScore)
            file_name = node.node.metadata.get("file_name")
            index_str = f"[{i}] " if file_name is None else f"[{i}] {file_name}"
            formatted_response += f"{index_str}\n\n{node.get_text()}\n"
        # print(formatted_response)
        return formatted_response

    def add_temporary_document(self, doc: BytesIO):
        """Add documents to the session store. Documents are a dictionary of document ID to base64 url"""
        if doc.name is None or doc.name == "":
            raise ValueError("Document name must be provided")
        assert isinstance(doc.name, str)
        # write the file to disk
        temp_file = self.__persist_dir / "temp" / doc.name
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.__persist_dir / "temp" / doc.name, "wb") as f:
            f.write(doc.read())
            f.flush()
        # index the file
        docs = SimpleDirectoryReader(
            input_files=[str(temp_file)], exclude_hidden=True
        ).load_data()
        store = self.vector_stores["default"]
        for d in docs:
            store.index.insert(d)
        # remove the temp file
        temp_file.unlink()

    def add_temporary_documents(self, docs: list[BytesIO]):
        for doc in docs:
            self.add_temporary_document(doc)
