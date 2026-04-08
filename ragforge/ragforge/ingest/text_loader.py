"""RAGForge — plain text file loader (TXT, MD, CSV, etc.)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragforge.core.types import Document, SourceType
from ragforge.ingest.base import BaseLoader


class TextLoader(BaseLoader):
    """Load plain text files."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".tsv", ".log", ".json", ".yaml", ".yml", ".xml", ".html", ".rst"}

    def load(self, source: str, **kwargs: Any) -> list[Document]:
        path = Path(source)
        metadata = kwargs.get("metadata", {})

        encoding = kwargs.get("encoding", "utf-8")
        try:
            content = path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1")

        if not content.strip():
            return []

        doc = Document(
            content=content,
            source=str(path.resolve()),
            source_type=SourceType.FILE,
            metadata={
                "filename": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                **metadata,
            },
        )
        return [doc]
