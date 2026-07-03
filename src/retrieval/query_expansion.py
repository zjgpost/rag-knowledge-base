"""Pseudo-relevance feedback query expansion.

Expands a query by appending high-frequency terms from the top-k BM25
retrieved documents, excluding terms already present in the query.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional

from retrieval.sparse_search import SparseSearch


class PRFQueryExpander:
    """Lightweight PRF query expansion using a first-stage sparse searcher."""

    def __init__(
        self,
        sparse_searcher: SparseSearch,
        top_k: int = 3,
        num_terms: int = 5,
    ):
        self.sparse = sparse_searcher
        self.top_k = top_k
        self.num_terms = num_terms

    def expand(self, query: str) -> str:
        if not self.sparse.bm25:
            return query

        results = self.sparse.search(query, top_k=self.top_k)
        if not results:
            return query

        query_tokens = set(SparseSearch._tokenize(query))
        term_counts: Counter = Counter()
        for doc_id, _ in results:
            for term in self.sparse.tokenized_docs[doc_id]:
                if term not in query_tokens:
                    term_counts[term] += 1

        top_terms = [term for term, _ in term_counts.most_common(self.num_terms)]
        if not top_terms:
            return query
        return f"{query} {' '.join(top_terms)}"
