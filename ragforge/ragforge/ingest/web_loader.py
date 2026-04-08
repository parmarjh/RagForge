"""RAGForge — web page loader."""

from __future__ import annotations

from typing import Any

from ragforge.core.types import Document, SourceType
from ragforge.ingest.base import BaseLoader


class WebLoader(BaseLoader):
    """Load text content from web pages using requests + BeautifulSoup."""

    def load(self, source: str, **kwargs: Any) -> list[Document]:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "requests and beautifulsoup4 are required for web loading. "
                "Install with: pip install requests beautifulsoup4"
            )

        metadata = kwargs.get("metadata", {})
        headers = kwargs.get("headers", {"User-Agent": "RAGForge/1.0"})
        timeout = kwargs.get("timeout", 30)

        response = requests.get(source, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Extract main content
        main = soup.find("main") or soup.find("article") or soup.find("body")
        text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        if not clean_text:
            return []

        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else source

        return [
            Document(
                content=clean_text,
                source=source,
                source_type=SourceType.URL,
                metadata={
                    "url": source,
                    "title": title_text,
                    "content_length": len(clean_text),
                    **metadata,
                },
            )
        ]
