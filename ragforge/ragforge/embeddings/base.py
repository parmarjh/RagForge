"""RAGForge — abstract base embedding provider."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a vector."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Default: sequential calls to embed()."""
        return [self.embed(t) for t in texts]

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimension size."""
        ...
