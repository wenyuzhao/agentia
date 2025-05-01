from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import NodeWithScore
import requests
from agentia.plugins.knowledge_base.vector_store import (
    SUPPORTED_CONTENT_TYPES,
    VectorStore,
    get_ext_from_content_type,
    is_file_supported,
)
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from slugify import slugify

from agentia.agent import Agent

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-4.1-mini")
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")


@dataclass
class Item:
    text: str
    score: float | None = None
    file_name: str | None = None

    def to_dict(self) -> dict[str, str | float | None]:
        return asdict(self)


class KnowledgeBase:
    def __init__(self, global_docs: Path | None = None):
        """
        Create or load a knowledge base.
        It will load vector stores from both a global store and a session store (if provided).
        """

        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY must be set to enable the knowledge base")

        self.vector_stores: dict[str, VectorStore] = {}
        # Load global vector store
        if global_docs is not None:
            assert global_docs.exists(), "Global documents directory does not exist"
            assert global_docs.is_dir(), "Global documents path is not a directory"
            self.vector_stores["global"] = VectorStore(
                name="global",
                persist_path=global_docs / ".vector_store",
                docs=global_docs,
            )
        self.__retriever: QueryFusionRetriever | None = None
        self.__session_store_path: Path | None = None
        self.__pending_documents: list[BytesIO | Path] = []

    def __init_retriever(self):
        from . import TOP_K

        retrievers = []
        for store in self.vector_stores.values():
            rs = store.as_retrievers(similarity_top_k=TOP_K)
            retrievers += rs
        if len(retrievers) == 0:
            self.__retriever = None
            return
        retriever = QueryFusionRetriever(
            retrievers,
            similarity_top_k=TOP_K,
            num_queries=4,  # set this to 1 to disable query generation
            mode=FUSION_MODES.RECIPROCAL_RANK,
            use_async=True,
            verbose=False,
        )
        self.__retriever = retriever

    def _create_session_store(self, agent: Agent):
        assert agent is not None
        session_store = agent.session_data_folder / "knowledge-base"
        session_store.mkdir(parents=True, exist_ok=True)
        assert "session" not in self.vector_stores, "Session store already exists"
        self.vector_stores["session"] = VectorStore(
            name="session", persist_path=session_store / "vector_store"
        )
        self.__session_store_path = session_store
        self.__init_retriever()
        docs = self.__pending_documents
        self.__pending_documents = []
        self.add_documents(docs)

    @staticmethod
    def is_file_supported(file_ext: str) -> bool:
        return is_file_supported(file_ext)

    async def query(self, query: str) -> list[Item]:
        """Query the knowledge base"""
        if self.__retriever is None:
            return []
        nodse_with_scores = await self.__retriever.aretrieve(query)
        result = []
        for node in nodse_with_scores:
            assert isinstance(node, NodeWithScore)
            result.append(
                Item(
                    score=node.score,
                    text=node.text,
                    file_name=node.node.metadata.get("file_name"),
                )
            )
        return result

    def __add_document(self, doc: BytesIO | Path):
        """Add documents to the session store. Documents are a dictionary of document ID to base64 url"""
        if self.__session_store_path is None:
            self.__pending_documents.append(doc)
            return
        if doc.name is None or doc.name == "":
            raise ValueError("Document name must be provided")
        assert isinstance(doc.name, str)
        if isinstance(doc, Path):
            # read to bytesio
            with open(doc, "rb") as f:
                doc = BytesIO(f.read())
        # write the file to disk
        assert self.__session_store_path is not None
        temp_file = self.__session_store_path / "docs" / doc.name
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(doc.read())
            f.flush()
        # index the file
        docs = SimpleDirectoryReader(
            input_files=[str(temp_file)], exclude_hidden=True
        ).load_data()
        store = self.vector_stores["session"]
        print(f"INSERTING {len(docs)} docs")
        for d in docs:
            store.insert(d)

    def add_document(self, doc: BytesIO | Path):
        """Add documents to the session store. Documents are a dictionary of document ID to base64 url"""
        self.__add_document(doc)
        self.__init_retriever()

    def add_documents(self, docs: list[BytesIO | Path]):
        if self.__session_store_path is None:
            self.__pending_documents += docs
            return
        for doc in docs:
            self.__add_document(doc)
        self.__init_retriever()

    def add_doc_from_url(self, url: str | requests.Response) -> str | None:
        """Add a document from a url"""
        res = url if isinstance(url, requests.Response) else requests.get(url)
        res.raise_for_status()
        content_type = res.headers.get("content-type")
        if content_type not in SUPPORTED_CONTENT_TYPES:
            return None
        # Add this file to the knowledge base
        data = BytesIO(res.content)
        # Get filename
        if s := res.headers.get("content-disposition"):
            data.name = s.split("filename=")[-1].strip('"')
        elif s := res.headers.get("x-filename"):
            data.name = s.strip('"')
        else:
            f = res.url.split("/")[-1]
            segments = [s.strip() for s in f.split(".")]
            data.name = ".".join(
                slugify(s, allow_unicode=True) for s in segments if s != ""
            )
            if Path(f).suffix == "":
                ext = get_ext_from_content_type(content_type)
                if ext is not None:
                    data.name += "." + ext
        # deduplicate the file name
        original_stem = Path(data.name).stem
        ext = Path(data.name).suffix
        suffix = 1
        assert self.__session_store_path is not None
        p = self.__session_store_path / "docs" / data.name
        while p.exists():
            suffix += 1
            p = self.__session_store_path / "docs" / f"{original_stem}-{suffix}.{ext}"
        data.name = p.name
        # Add to knowledge base
        self.add_document(data)
        return data.name
