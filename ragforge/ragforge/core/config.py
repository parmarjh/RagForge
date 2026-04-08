"""RAGForge — configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ragforge.core.types import ChunkStrategy


@dataclass
class EmbeddingConfig:
    provider: str = "openai"  # "openai", "huggingface", "ollama"
    model: str = "text-embedding-3-small"
    api_key: str | None = None
    base_url: str | None = None
    dimensions: int = 1536
    batch_size: int = 100
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    provider: str = "openai"  # "openai", "anthropic", "ollama"
    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.1
    max_tokens: int = 2048
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorStoreConfig:
    provider: str = "qdrant"
    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection: str = "ragforge"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkConfig:
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )


@dataclass
class RetrievalConfig:
    top_k: int = 5
    score_threshold: float = 0.0
    rerank: bool = False
    hybrid: bool = False  # combine vector + keyword search
    hybrid_alpha: float = 0.7  # weight for vector vs keyword (1.0 = pure vector)


@dataclass
class RAGConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
