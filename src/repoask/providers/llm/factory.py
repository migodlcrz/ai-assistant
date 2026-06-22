from __future__ import annotations

from ...config.schema import LLMConfig
from ..base import LLMProvider


def create_llm_provider(config: LLMConfig) -> LLMProvider:
    if config.provider == "ollama":
        from .ollama import OllamaLLMProvider
        return OllamaLLMProvider(
            model=config.model,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    if not config.api_key:
        raise SystemExit(
            f"API key is required for LLM provider '{config.provider}'.\n"
            "Run: repoask config  to set it up."
        )

    if config.provider == "groq":
        from .groq import GroqLLMProvider
        return GroqLLMProvider(
            model=config.model,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    if config.provider == "openai":
        from .openai import OpenAILLMProvider
        return OpenAILLMProvider(
            model=config.model,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    if config.provider == "anthropic":
        from .anthropic import AnthropicLLMProvider
        return AnthropicLLMProvider(
            model=config.model,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    raise SystemExit(f"Unknown LLM provider: {config.provider!r}")
