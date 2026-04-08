"""RAGForge — abstract base loader."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ragforge.core.types import Document


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, source: str, **kwargs: Any) -> list[Document]:
        """Load documents from a source. Returns a list of Document objects."""
        ...
