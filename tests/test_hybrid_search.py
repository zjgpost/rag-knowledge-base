"""Tests for hybrid search."""

import pytest

from retrieval.hybrid_search import HybridSearch


@pytest.fixture
def searcher():
    docs = [
        "Redis connection pool max_connections controls the maximum number of connections.",
        "Redis performance optimization includes tuning memory and persistence.",
        "How to monitor CPU and memory usage on Linux servers.",
    ]
    h = HybridSearch(dense_weight=0.7, sparse_weight=0.3)
    h.fit(docs)
    return h


def test_hybrid_search_returns_results(searcher):
    results = searcher.search("Redis connection pool max connections")
    assert len(results) > 0
    assert "Redis" in results[0]["content"]


def test_dynamic_weights(searcher):
    searcher.set_weights(0.8, 0.2)
    assert abs(searcher.dense_weight - 0.8) < 0.01
    assert abs(searcher.sparse_weight - 0.2) < 0.01
