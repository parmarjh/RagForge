"""RAGForge — abstract base vector store."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragforge.core.types import Chunk, SearchResult


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or update chunks with their embeddings."""
        ...

    @abstractmethod
    def search(
        self, embedding: list[float], top_k: int = 5, score_threshold: float = 0.0
    ) -> list[SearchResult]:
        """Search for similar chunks by embedding vector."""
        ...

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by their IDs."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored chunks."""
        ...

    def delete_collection(self) -> None:
        """Delete the entire collection. Override in subclasses."""
        raise NotImplementedError
