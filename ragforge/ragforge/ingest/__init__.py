"""RAGForge ingest package — document loaders and chunking."""

from ragforge.ingest.base import BaseLoader
from ragforge.ingest.chunker import TextChunker

__all__ = ["BaseLoader", "TextChunker"]
