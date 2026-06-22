from __future__ import annotations

import hashlib
from pathlib import Path

from ..ingestion.chunker import Chunk

COLLECTION_NAME = "repoask_chunks"


class VectorStore:
    def __init__(self, store_dir: Path):
        try:
            import chromadb
        except ImportError:
            raise SystemExit("chromadb is required: pip install chromadb")

        store_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(store_dir))
        self._col = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #

    def _chunk_id(self, chunk: Chunk) -> str:
        key = f"{chunk.file_path}:{chunk.start_line}:{chunk.end_line}:{chunk.symbol_name}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        ids = [self._chunk_id(c) for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "file_path": c.file_path,
                "language": c.language,
                "symbol_name": c.symbol_name,
                "symbol_type": c.symbol_type,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "imports": ",".join(c.imports),
            }
            for c in chunks
        ]
        self._col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def delete_by_file(self, file_path: str) -> None:
        results = self._col.get(where={"file_path": file_path})
        if results and results["ids"]:
            self._col.delete(ids=results["ids"])

    def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        kwargs: dict = {
            "query_embeddings": [embedding],
            "n_results": min(top_k, max(self._col.count(), 1)),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._col.query(**kwargs)

        hits: list[dict] = []
        for i, doc_id in enumerate(results["ids"][0]):
            hits.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return hits

    def count(self) -> int:
        return self._col.count()

    def stats(self) -> dict:
        count = self._col.count()
        if count == 0:
            return {"chunks": 0, "files": 0}
        all_meta = self._col.get(include=["metadatas"])["metadatas"]
        files = {m["file_path"] for m in all_meta}
        return {"chunks": count, "files": len(files)}
