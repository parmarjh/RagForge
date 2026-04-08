"""RAGForge — prompt templates for RAG pipeline."""

from __future__ import annotations


QA_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Rules:
- Only use information from the provided context to answer
- If the context doesn't contain enough information, say so clearly
- Cite which parts of the context support your answer
- Be concise and direct"""

QA_USER_PROMPT = """Context:
{context}

---

Question: {question}

Answer based on the context above:"""


SUMMARIZE_SYSTEM_PROMPT = """You are a precise summarizer. Summarize the provided text concisely while preserving key information."""

SUMMARIZE_USER_PROMPT = """Summarize the following text:

{text}

Summary:"""


CONDENSE_QUESTION_PROMPT = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question that captures the full context.

Chat history:
{chat_history}

Follow-up question: {question}

Standalone question:"""


def format_context(chunks_text: list[str], max_chars: int = 8000) -> str:
    """Format retrieved chunks into a context string."""
    context_parts = []
    total = 0

    for i, text in enumerate(chunks_text):
        entry = f"[Source {i + 1}]\n{text}\n"
        if total + len(entry) > max_chars:
            break
        context_parts.append(entry)
        total += len(entry)

    return "\n".join(context_parts)
