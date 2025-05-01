import json
from pathlib import Path
import shelve
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import SimpleDirectoryReader
from filelock import FileLock
import logging
from llama_index.core import SimpleDirectoryReader, Document


def is_file_supported(file_ext: str) -> bool:
    return file_ext.lower().strip(".") in SUPPORTED_EXTS


class VectorStore:
    def __init__(self, name: str, persist_path: Path, docs: Path | None = None):
        """
        Initialize a vector store. If the given path already exists, it will be loaded.
        The contents of the `docs` directory will be indexed. Any docs that are not in the directory will be removed from the index.

        :param persist_path: Base path to store the vector store
        :param docs: Optional path to the directory containing the documents. Default to <persist_path>/docs`
        """
        self.name = name
        self.__persist_path = persist_path
        self.__persist_path.mkdir(parents=True, exist_ok=True)
        self.__client = chromadb.PersistentClient(
            path=str(persist_path / "vector_store")
        )
        self.__vector_store = ChromaVectorStore(
            chroma_collection=self.__client.get_or_create_collection("vector_store")
        )
        self.__index = VectorStoreIndex.from_vector_store(self.__vector_store)
        docs = docs or self.__persist_path / "docs"
        docs.mkdir(parents=True, exist_ok=True)
        if docs:
            self.__update_from_source(docs)

    def get_all_nodes(self) -> dict[str, list[str]]:
        results = self.__vector_store._collection.get(limit=None)
        nodes = {}
        metadatas = results.get("metadatas", [])
        for m in metadatas:  # type: ignore
            if nc := m.get("_node_content", {}):
                json_data = json.loads(nc)  # type: ignore
                id_ = json_data["id_"]
                if f := json_data.get("metadata", {}).get("file_name"):  # type: ignore
                    if f not in nodes:
                        nodes[f] = []
                    nodes[f].append(id_)
        return nodes

    def get_indexed_files(self) -> list[str]:
        results = self.__vector_store._collection.get(limit=None)
        files = set()
        metadatas = results.get("metadatas", [])
        for m in metadatas:  # type: ignore
            if nc := m.get("_node_content", {}):
                json_data = json.loads(nc)  # type: ignore
                if f := json_data.get("metadata", {}).get("file_name"):  # type: ignore
                    files.add(f)
        return list(files)

    def insert(self, doc: Document):
        assert doc.metadata.get("file_name") is not None
        self.__index.insert(doc)

    def __update_from_source(self, source: Path) -> list[str]:
        self.__persist_path.mkdir(parents=True, exist_ok=True)
        indexed_files_path = self.__persist_path / "indexed_files"
        lock_path = self.__persist_path / "lock"
        # Collect all the files
        files = {
            f.name: f
            for f in source.iterdir()
            if f.is_file()
            and is_file_supported(f.suffix)
            and not f.name.startswith(".")
            and not f.name.startswith("_")
        }
        all_files = list(files.keys())
        del_files: set[str] = set()
        with FileLock(lock_path):
            with shelve.open(indexed_files_path) as g:
                for f in g:
                    if f not in files:
                        del_files.add(f)
                    elif g[f] >= files[f].stat().st_mtime:  # not modified
                        del files[f]
                    else:  # file is modified
                        del_files.add(f)
            # Remove files
            if len(del_files) > 0:
                nodes = self.get_all_nodes()
                del_files_set = set(del_files)
                nodes_to_remove = []
                for k, v in nodes.items():
                    if k in del_files_set:
                        nodes_to_remove += v
                self.__index.delete_nodes(nodes_to_remove)
            # Index new files
            if len(files) > 0:
                logging.info(f"Indexing {len(files)} files")
                docs = SimpleDirectoryReader(
                    input_files=[str(f) for f in files.values()],
                    exclude_hidden=False,
                ).load_data()
                for d in docs:
                    self.__index.insert(d)
            # Update the global files
            with shelve.open(indexed_files_path) as g:
                for f in del_files:
                    del g[f]
                for f in files:
                    g[f] = files[f].stat().st_mtime
        return all_files

    def as_retrievers(self, similarity_top_k: int):
        a = self.__index.as_retriever(similarity_top_k=similarity_top_k)
        # b = BM25Retriever.from_defaults(
        #     docstore=self.__index.docstore, similarity_top_k=similarity_top_k
        # )
        return [a]


SUPPORTED_EXTS = [
    "csv",
    "docx",
    "epub",
    "hwp",
    "ipynb",
    "jpeg",
    "jpg",
    "mbox",
    "md",
    "mp3",
    "mp4",
    "pdf",
    "png",
    "ppt",
    "pptm",
    "pptx",
    # common text files
    "txt",
    "log",
    "tex",
]

SUPPORTED_CONTENT_TYPES = [
    "application/pdf",
    "text/csv",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/epub+zip",
    "application/x-hwp",
    "application/x-ipynb+json",
    "image/jpeg",
    "image/jpg",
    "text/x-mbox",
    "text/markdown",
    "audio/mpeg",
    "video/mp4",
    "image/png",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.presentationml.slide",
    "application/vnd.openxmlformats-officedocument.presentationml.template",
    "text/plain",
    "text/html",
]


def get_ext_from_content_type(ct: str) -> str | None:
    """
    Get the file extension from the content type
    """
    match ct:
        case "application/pdf":
            return "pdf"
        case "text/csv":
            return "csv"
        case "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return "docx"
        case "application/epub+zip":
            return "epub"
        case "application/x-hwp":
            return "hwp"
        case "application/x-ipynb+json":
            return "ipynb"
        case "image/jpeg" | "image/jpg":
            return "jpg"
        case "text/x-mbox":
            return "mbox"
        case "text/markdown":
            return "md"
        case "audio/mpeg":
            return "mp3"
        case "video/mp4":
            return "mp4"
        case "image/png":
            return "png"
        case (
            "application/vnd.ms-powerpoint"
            | "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            | "application/vnd.openxmlformats-officedocument.presentationml.slide"
            | "application/vnd.openxmlformats-officedocument.presentationml.template"
        ):
            return "pptx"
        case "text/plain":
            return "txt"
        case "text/html":
            return "html"
        case _:
            return None
