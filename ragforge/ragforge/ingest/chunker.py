"""RAGForge — text chunking strategies."""

from __future__ import annotations

from ragforge.core.types import Chunk, ChunkStrategy, Document


class TextChunker:
    """Split documents into chunks using configurable strategies."""

    def __init__(
        self,
        strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split(self, document: Document) -> list[Chunk]:
        """Split a document into chunks."""
        text = document.content.strip()
        if not text:
            return []

        if self.strategy == ChunkStrategy.FIXED:
            texts = self._fixed_split(text)
        elif self.strategy == ChunkStrategy.RECURSIVE:
            texts = self._recursive_split(text, self.separators)
        elif self.strategy == ChunkStrategy.SEMANTIC:
            texts = self._semantic_split(text)
        else:
            texts = self._recursive_split(text, self.separators)

        chunks = []
        for i, t in enumerate(texts):
            if t.strip():
                chunks.append(
                    Chunk(
                        text=t.strip(),
                        doc_id=document.doc_id,
                        chunk_index=i,
                        metadata={
                            "source": document.source,
                            "source_type": document.source_type.value,
                            **document.metadata,
                        },
                    )
                )
        return chunks

    def _fixed_split(self, text: str) -> list[str]:
        """Split text into fixed-size chunks with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using a hierarchy of separators."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Find the best separator
        sep = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else [""]

        if sep:
            parts = text.split(sep)
        else:
            # Character-level split as last resort
            return self._fixed_split(text)

        chunks = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If single part exceeds size, split recursively with next separator
                if len(part) > self.chunk_size:
                    sub_chunks = self._recursive_split(part, remaining_seps)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        # Apply overlap by prepending tail of previous chunk
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_tail = chunks[i - 1][-self.chunk_overlap :]
                overlapped.append(prev_tail + " " + chunks[i])
            return overlapped

        return chunks

    def _semantic_split(self, text: str) -> list[str]:
        """Split by semantic boundaries (paragraphs, then sentences).

        A lightweight heuristic: split on double-newlines first,
        then merge small paragraphs or split large ones.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 <= self.chunk_size:
                current = current + "\n\n" + para if current else para
            else:
                if current:
                    chunks.append(current)
                if len(para) > self.chunk_size:
                    # Split long paragraph by sentences
                    sentences = para.replace(". ", ".\n").split("\n")
                    sub = ""
                    for s in sentences:
                        if len(sub) + len(s) + 1 <= self.chunk_size:
                            sub = sub + " " + s if sub else s
                        else:
                            if sub:
                                chunks.append(sub)
                            sub = s
                    if sub:
                        current = sub
                    else:
                        current = ""
                else:
                    current = para

        if current:
            chunks.append(current)

        return chunks
