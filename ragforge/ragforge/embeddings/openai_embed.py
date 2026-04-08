"""RAGForge — OpenAI embedding provider."""

from __future__ import annotations

import os
from typing import Any

from ragforge.embeddings.base import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    """Generate embeddings using OpenAI's API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int = 1536,
        batch_size: int = 100,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

        self._model = model
        self._dims = dimensions
        self._batch_size = batch_size
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            model=self._model,
            input=text,
            dimensions=self._dims,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dims,
            )
            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([d.embedding for d in sorted_data])
        return all_embeddings

    @property
    def dimensions(self) -> int:
        return self._dims
