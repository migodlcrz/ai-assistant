from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    provider: Literal["huggingface", "openai"] = "huggingface"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    api_key: str = ""


class LLMConfig(BaseModel):
    provider: Literal["groq", "openai", "anthropic"] = "groq"
    model: str = "llama3-8b-8192"
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 2048


class IndexingConfig(BaseModel):
    ignore_patterns: list[str] = Field(
        default=[
            ".git", "node_modules", "__pycache__", "dist", "build",
            ".venv", "venv", "*.lock", "*.min.js", "*.min.css",
            "*.pyc", ".DS_Store", "coverage", ".next", ".nuxt",
        ]
    )
    max_file_size_kb: int = 500
    languages: list[str] = Field(default=["python", "javascript", "typescript"])


class StoreConfig(BaseModel):
    path: str = ".repoask"


class RepoAskConfig(BaseModel):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
