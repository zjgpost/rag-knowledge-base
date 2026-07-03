"""Tests for document chunker."""

import pytest

from ingestion.chunker import chunk_fixed, chunk_recursive, chunk_semantic, chunk_text


def test_chunk_fixed():
    text = "a" * 1000
    chunks = chunk_fixed(text, chunk_size=300, overlap=50)
    assert len(chunks) >= 3
    assert len(chunks[0]) == 300


def test_chunk_semantic_preserves_sentences():
    text = "First sentence. Second sentence. Third sentence."
    chunks = chunk_semantic(text, max_chunk_size=200)
    assert len(chunks) == 1
    assert "First sentence" in chunks[0]


def test_chunk_recursive():
    text = "Paragraph one.\n\nParagraph two with more words.\n\nParagraph three."
    chunks = chunk_recursive(text, max_chunk_size=200)
    assert len(chunks) >= 1


def test_chunk_text_routing():
    text = "A. B. C."
    assert chunk_text(text, strategy="semantic") == chunk_semantic(text)
    assert chunk_text(text, strategy="fixed", chunk_size=100) == chunk_fixed(text, 100)
