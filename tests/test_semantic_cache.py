"""Tests for semantic cache."""

import pytest

from cache.semantic_cache import SemanticCache


@pytest.fixture
def cache():
    return SemanticCache(similarity_threshold=0.5)


def test_cache_hit(cache):
    cache.set("Redis connection pool full", "Increase max_connections.")
    hit, entry, score = cache.get("Redis connection pool is full")
    assert hit
    assert entry["answer"] == "Increase max_connections."


def test_cache_miss(cache):
    cache.set("Redis connection pool full", "Increase max_connections.")
    hit, _, _ = cache.get("How to optimize PostgreSQL")
    assert not hit


def test_cache_stats(cache):
    cache.set("q1", "a1")
    cache.set("q2", "a2")
    assert cache.stats()["entries"] == 2
