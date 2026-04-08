"""RAGForge — OpenAI LLM provider."""

from __future__ import annotations

import os

from ragforge.llm.base import BaseLLM, LLMResponse


class OpenAILLM(BaseLLM):
    """Generate responses using OpenAI's GPT models."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = OpenAI(api_key=key, base_url=base_url)
        self._aclient = AsyncOpenAI(api_key=key, base_url=base_url)

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            text=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    async def agenerate(self, prompt: str, system: str | None = None) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._aclient.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            text=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
