from __future__ import annotations

from ..base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str, api_key: str):
        try:
            from openai import OpenAI
        except ImportError:
            raise SystemExit("openai package is required: pip install openai")
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]
