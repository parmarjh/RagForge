"""RAGForge — SQL database loader."""

from __future__ import annotations

from typing import Any

from ragforge.core.types import Document, SourceType
from ragforge.ingest.base import BaseLoader


class DatabaseLoader(BaseLoader):
    """Load text content from SQL database queries using SQLAlchemy."""

    def load(self, source: str, **kwargs: Any) -> list[Document]:
        try:
            from sqlalchemy import create_engine, text
        except ImportError:
            raise ImportError(
                "SQLAlchemy is required for database loading. "
                "Install with: pip install sqlalchemy"
            )

        query = kwargs.get("query")
        if not query:
            raise ValueError("A SQL query is required. Pass query='SELECT ...'")

        text_column = kwargs.get("text_column", "content")
        metadata_columns = kwargs.get("metadata_columns")  # optional list
        metadata = kwargs.get("metadata", {})

        engine = create_engine(source)
        documents = []

        with engine.connect() as conn:
            result = conn.execute(text(query))
            columns = list(result.keys())

            for i, row in enumerate(result):
                row_dict = dict(zip(columns, row))

                # Extract text content
                content = str(row_dict.get(text_column, ""))
                if not content.strip():
                    # Fall back to joining all string columns
                    content = " ".join(
                        str(v) for v in row_dict.values() if isinstance(v, str) and v.strip()
                    )

                if not content.strip():
                    continue

                # Build metadata from other columns
                row_meta = {}
                cols_to_include = metadata_columns or [c for c in columns if c != text_column]
                for col in cols_to_include:
                    val = row_dict.get(col)
                    if val is not None and isinstance(val, (str, int, float, bool)):
                        row_meta[col] = val

                documents.append(
                    Document(
                        content=content,
                        source=source.split("@")[-1] if "@" in source else source,
                        source_type=SourceType.DATABASE,
                        metadata={
                            "row_index": i,
                            "query": query[:200],
                            **row_meta,
                            **metadata,
                        },
                    )
                )

        engine.dispose()
        return documents
