"""Sparse BM25 retrieval."""

from __future__ import annotations

import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi


_PLACEHOLDER_RE = re.compile(r"\[([A-Z_]+)\]")


class SparseSearch:
    """BM25 keyword retrieval."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[str] = []
        self.tokenized_docs: List[List[str]] = []
        self.bm25: BM25Okapi | None = None

    @staticmethod
    def _normalize(text: str) -> str:
        # Make placeholders like [ORDER_ID] searchable as ORDER_ID.
        return _PLACEHOLDER_RE.sub(r" \1 ", text)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_]+\b", SparseSearch._normalize(text).lower())

    def fit(self, documents: List[str]) -> None:
        self.documents = documents
        self.tokenized_docs = [self._tokenize(d) for d in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.bm25 is None or not self.documents:
            return []
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(i, float(scores[i])) for i in top_indices if scores[i] > 0]
