from __future__ import annotations

from ...config.schema import EmbeddingConfig
from ..base import EmbeddingProvider


def create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    if config.provider == "huggingface":
        from .huggingface import HuggingFaceEmbeddingProvider
        return HuggingFaceEmbeddingProvider(model=config.model)

    if config.provider == "openai":
        if not config.api_key:
            raise SystemExit("OpenAI API key is required for openai embedding provider.")
        from .openai import OpenAIEmbeddingProvider
        return OpenAIEmbeddingProvider(model=config.model, api_key=config.api_key)

    raise SystemExit(f"Unknown embedding provider: {config.provider!r}")
