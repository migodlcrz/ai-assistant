from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: list[Message], stream: bool = False):
        """
        Non-streaming: return full response string.
        Streaming: yield string chunks.
        """
