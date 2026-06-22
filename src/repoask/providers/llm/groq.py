from __future__ import annotations

from typing import Generator

from ..base import LLMProvider, Message


class GroqLLMProvider(LLMProvider):
    def __init__(self, model: str, api_key: str, temperature: float, max_tokens: int):
        try:
            from groq import Groq
        except ImportError:
            raise SystemExit("groq package is required: pip install groq")
        self._client = Groq(api_key=api_key)
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
