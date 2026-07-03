"""Document chunking strategies."""

from __future__ import annotations

import re
from typing import List, Literal


ChunkStrategy = Literal["fixed", "semantic", "recursive"]


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using simple punctuation heuristics."""
    sentences = re.split(r"(?<=[。！？.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_fixed(text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
    """Fixed-size chunking with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start <= 0:
            start = end
    return chunks


def chunk_semantic(text: str, max_chunk_size: int = 512) -> List[str]:
    """Semantic chunking at sentence boundaries."""
    sentences = split_sentences(text)
    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > max_chunk_size and current:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_recursive(text: str, max_chunk_size: int = 512) -> List[str]:
    """Recursive chunking: paragraphs -> sentences -> fixed pieces."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []

    for paragraph in paragraphs:
        if len(paragraph) <= max_chunk_size:
            chunks.append(paragraph)
            continue

        sentences = split_sentences(paragraph)
        current = []
        current_len = 0
        for sentence in sentences:
            if current_len + len(sentence) > max_chunk_size and current:
                chunks.append(" ".join(current))
                current = [sentence]
                current_len = len(sentence)
            else:
                current.append(sentence)
                current_len += len(sentence)
        if current:
            chunks.append(" ".join(current))

    return chunks


def chunk_text(text: str, strategy: ChunkStrategy = "semantic", **kwargs) -> List[str]:
    """Route to the requested chunking strategy."""
    if strategy == "fixed":
        return chunk_fixed(text, kwargs.get("chunk_size", 512), kwargs.get("overlap", 128))
    if strategy == "semantic":
        return chunk_semantic(text, kwargs.get("max_chunk_size", 512))
    if strategy == "recursive":
        return chunk_recursive(text, kwargs.get("max_chunk_size", 512))
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def auto_chunk(text: str, doc_type: str = "text") -> List[str]:
    """Choose a chunking strategy based on document type."""
    doc_type = doc_type.lower()
    if doc_type in ("markdown", "html", "pdf"):
        return chunk_semantic(text)
    if doc_type in ("legal", "contract"):
        return chunk_recursive(text)
    return chunk_fixed(text)
