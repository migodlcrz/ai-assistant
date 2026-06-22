from __future__ import annotations

from ..base import EmbeddingProvider


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise SystemExit(
                "sentence-transformers is required for HuggingFace embeddings.\n"
                "Install it with: pip install sentence-transformers"
            )
        self._model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vectors]
