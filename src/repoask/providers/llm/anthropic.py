from __future__ import annotations

from typing import Generator

from ..base import LLMProvider, Message


class AnthropicLLMProvider(LLMProvider):
    def __init__(self, model: str, api_key: str, temperature: float, max_tokens: int):
        try:
            import anthropic
        except ImportError:
            raise SystemExit("anthropic package is required: pip install anthropic")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def chat(self, messages: list[Message], stream: bool = False):
        system_msg = ""
        user_messages = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                user_messages.append({"role": m.role, "content": m.content})

        if stream:
            return self._stream(system_msg, user_messages)

        response = self._client.messages.create(
            model=self._model,
            system=system_msg,
            messages=user_messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.content[0].text

    def _stream(self, system_msg: str, messages: list[dict]) -> Generator[str, None, None]:
        with self._client.messages.stream(
            model=self._model,
            system=system_msg,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        ) as stream:
            for text in stream.text_stream:
                yield text
