"""RAGForge — shared types and data structures."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SourceType(str, Enum):
    FILE = "file"
    URL = "url"
    API = "api"
    DATABASE = "database"
    TEXT = "text"


class ChunkStrategy(str, Enum):
    RECURSIVE = "recursive"
    FIXED = "fixed"
    SEMANTIC = "semantic"


@dataclass
class Document:
    """A raw document before chunking."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    source_type: SourceType = SourceType.TEXT
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Chunk:
    """A chunk of text extracted from a document."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    chunk_index: int = 0
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    chunk: Chunk
    score: float = 0.0
    rank: int = 0


@dataclass
class GenerationResult:
    """The final output from a RAG query."""

    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    confidence: float = 0.0
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
