"""
Chroma-backed retriever for the bonus notebook (section 07).

Exposes the same `.query()` shape as `utils/retriever.Retriever`, so the same
agent loop code runs unchanged. Backed by `ibm-granite/granite-embedding-30m-english`
(same embedding model as the EscalationLab) and a persisted Chroma collection at
`prebuilt/chroma_agentic/`.

Many workshop environments ship an older sqlite than Chroma requires, so we
swap in pysqlite3 before importing chromadb.
"""

import json
import sys
from pathlib import Path

try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import chromadb
from chromadb.utils import embedding_functions

CHUNKS_PATH = Path(__file__).parent.parent / "prebuilt" / "bfrpg_chunks.json"
CHROMA_PATH = Path(__file__).parent.parent / "prebuilt" / "chroma_agentic"
COLLECTION_NAME = "basic_fantasy_corpus_embeddings"
EMBEDDING_MODEL = "ibm-granite/granite-embedding-30m-english"


def _embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


class ChromaRetriever:
    """ChromaDB-compatible retriever matching utils.retriever.Retriever's shape."""

    def __init__(self, collection, chunks):
        self._collection = collection
        self.chunks = chunks

    @property
    def name(self):
        return self._collection.name

    def count(self):
        return self._collection.count()

    def query(self, query_texts, n_results=3, include=None):
        include = include or ["documents", "distances"]
        return self._collection.query(
            query_texts=query_texts,
            n_results=n_results,
            include=include,
        )

    @classmethod
    def load(cls, chroma_path=None, chunks_path=None, build_if_missing=True):
        chroma_path = Path(chroma_path) if chroma_path else CHROMA_PATH
        chunks_path = Path(chunks_path) if chunks_path else CHUNKS_PATH

        client = chromadb.PersistentClient(path=str(chroma_path))
        embed_fn = _embedding_fn()

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        existing = {c.name for c in client.list_collections()}
        if COLLECTION_NAME in existing:
            collection = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
            if collection.count() == len(chunks):
                return cls(collection, chunks)
            if not build_if_missing:
                return cls(collection, chunks)
            client.delete_collection(COLLECTION_NAME)

        if not build_if_missing:
            raise RuntimeError(
                f"Chroma collection '{COLLECTION_NAME}' not found at {chroma_path} "
                "and build_if_missing=False"
            )

        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"source": "bfrpg_chunks.json", "embedding_model": EMBEDDING_MODEL},
        )
        collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{"source_section": c.get("source_section", "")} for c in chunks],
        )
        return cls(collection, chunks)
