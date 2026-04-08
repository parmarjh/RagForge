"""RAGForge — PDF document loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragforge.core.types import Document, SourceType
from ragforge.ingest.base import BaseLoader


class PDFLoader(BaseLoader):
    """Load text content from PDF files using PyMuPDF (fitz)."""

    def load(self, source: str, **kwargs: Any) -> list[Document]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF loading. "
                "Install it with: pip install pymupdf"
            )

        path = Path(source)
        metadata = kwargs.get("metadata", {})
        per_page = kwargs.get("per_page", False)

        pdf = fitz.open(str(path))
        documents = []

        if per_page:
            for i, page in enumerate(pdf):
                text = page.get_text()
                if text.strip():
                    documents.append(
                        Document(
                            content=text,
                            source=str(path.resolve()),
                            source_type=SourceType.FILE,
                            metadata={
                                "filename": path.name,
                                "page": i + 1,
                                "total_pages": len(pdf),
                                **metadata,
                            },
                        )
                    )
        else:
            full_text = "\n\n".join(
                page.get_text() for page in pdf if page.get_text().strip()
            )
            if full_text.strip():
                documents.append(
                    Document(
                        content=full_text,
                        source=str(path.resolve()),
                        source_type=SourceType.FILE,
                        metadata={
                            "filename": path.name,
                            "total_pages": len(pdf),
                            **metadata,
                        },
                    )
                )

        pdf.close()
        return documents
