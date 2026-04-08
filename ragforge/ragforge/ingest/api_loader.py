"""RAGForge — REST API loader."""

from __future__ import annotations

import json
from typing import Any

from ragforge.core.types import Document, SourceType
from ragforge.ingest.base import BaseLoader


class APILoader(BaseLoader):
    """Load text content from REST API JSON endpoints."""

    def load(self, source: str, **kwargs: Any) -> list[Document]:
        try:
            import requests
        except ImportError:
            raise ImportError("requests is required. Install with: pip install requests")

        metadata = kwargs.get("metadata", {})
        method = kwargs.get("method", "GET").upper()
        headers = kwargs.get("headers") or {}
        params = kwargs.get("params")
        body = kwargs.get("body")
        json_path = kwargs.get("json_path", "")
        text_field = kwargs.get("text_field", "content")
        timeout = kwargs.get("timeout", 30)

        response = requests.request(
            method, source, headers=headers, params=params, json=body, timeout=timeout
        )
        response.raise_for_status()
        data = response.json()

        # Navigate to nested path if specified (e.g., "data.results")
        if json_path:
            for key in json_path.split("."):
                if isinstance(data, dict):
                    data = data[key]
                elif isinstance(data, list) and key.isdigit():
                    data = data[int(key)]

        # Handle both single objects and arrays
        items = data if isinstance(data, list) else [data]

        documents = []
        for i, item in enumerate(items):
            if isinstance(item, dict):
                text = item.get(text_field, "")
                if not text:
                    text = json.dumps(item, indent=2, ensure_ascii=False)
                item_meta = {k: v for k, v in item.items() if k != text_field and isinstance(v, (str, int, float, bool))}
            elif isinstance(item, str):
                text = item
                item_meta = {}
            else:
                text = str(item)
                item_meta = {}

            if text.strip():
                documents.append(
                    Document(
                        content=text,
                        source=source,
                        source_type=SourceType.API,
                        metadata={
                            "api_url": source,
                            "item_index": i,
                            **item_meta,
                            **metadata,
                        },
                    )
                )

        return documents
