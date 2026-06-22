from __future__ import annotations

from typing import Generator

from ..base import LLMProvider, Message


class OllamaLLMProvider(LLMProvider):
    def __init__(self, model: str, base_url: str = "http://localhost:11434", temperature: float = 0.2, max_tokens: int = 2048):
        try:
            from openai import OpenAI
        except ImportError:
            raise SystemExit("openai package is required for Ollama: pip install openai")
        # Ollama exposes an OpenAI-compatible API
        self._client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def chat(self, messages: list[Message], stream: bool = False):
        payload = [{"role": m.role, "content": m.content} for m in messages]

        if stream:
            return self._stream(payload)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=payload,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content

    def _stream(self, payload: list[dict]) -> Generator[str, None, None]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=payload,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
