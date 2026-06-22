from __future__ import annotations

from ..providers.base import EmbeddingProvider
from ..store.chroma import VectorStore


class Retriever:
    def __init__(self, store: VectorStore, embedder: EmbeddingProvider):
        self._store = store
        self._embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        language_filter: str | None = None,
        file_filter: str | None = None,
    ) -> list[dict]:
        embedding = self._embedder.embed_one(query)

        where: dict | None = None
        conditions = []
        if language_filter:
            conditions.append({"language": {"$eq": language_filter}})
        if file_filter:
            conditions.append({"file_path": {"$eq": file_filter}})

        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        return self._store.query(embedding=embedding, top_k=top_k, where=where)
