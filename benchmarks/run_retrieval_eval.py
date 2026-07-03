"""Reproducible benchmark for hybrid retrieval.

Compares:
- Dense-only retrieval
- Sparse-only retrieval
- Hybrid (Dense + Sparse + Rerank)

Metrics: Recall@K, NDCG@K
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieval.dense_search import DenseSearch
from retrieval.hybrid_search import HybridSearch
from retrieval.sparse_search import SparseSearch


DATASET_DIR = Path(__file__).parent / "dataset"

DOCS = [
    # 0: Redis connection pool
    "Redis connection pool max_connections controls the maximum number of client connections. "
    "When the pool is exhausted, new connections will be rejected. Increase max_connections or "
    "use connection pooling in your application to reuse connections.",
    # 1: Redis performance
    "Redis performance optimization involves tuning memory policies, persistence settings, and "
    "network parameters. Use INFO stats to identify bottlenecks and adjust maxmemory-policy accordingly.",
    # 2: Linux monitoring
    "Linux server monitoring uses tools like top, htop, vmstat, and iostat to observe CPU, memory, "
    "disk, and network utilization. Set up alerts when usage exceeds thresholds.",
    # 3: distractor — PostgreSQL
    "PostgreSQL connection pooling with PgBouncer reduces the overhead of creating new database "
    "connections. Set pool_mode to transaction or session depending on your workload.",
    # 4: distractor — Nginx
    "Nginx performance tuning includes worker_processes, worker_connections, and keepalive settings. "
    "Tuning these parameters improves web server throughput under high load.",
    # 5: distractor — Kubernetes
    "Kubernetes resource monitoring relies on metrics-server and Prometheus to observe pod CPU and "
    "memory usage. Use Horizontal Pod Autoscaler to scale based on metrics.",
    # 6: distractor — MySQL
    "MySQL max_connections defines the maximum number of simultaneous client connections. Too many "
    "connections can exhaust memory and cause the server to reject new clients.",
    # 7: distractor — Elasticsearch
    "Elasticsearch cluster performance depends on shard allocation, heap size, and query caching. "
    "Monitor node metrics to avoid memory pressure and slow queries.",
]


def load_queries() -> List[Dict[str, Any]]:
    with open(DATASET_DIR / "queries.jsonl", "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dcg(scores: List[float]) -> float:
    return sum((2 ** s - 1) / math.log2(i + 2) for i, s in enumerate(scores))


def ndcg_at_k(results: List[int], relevant_indices: List[int], k: int = 10) -> float:
    rel = [1 if r in relevant_indices else 0 for r in results[:k]]
    ideal = [1] * min(len(relevant_indices), k) + [0] * max(0, k - len(relevant_indices))
    actual_dcg = dcg(rel)
    ideal_dcg = dcg(ideal)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(results: List[int], relevant_indices: List[int], k: int = 10) -> float:
    hits = sum(1 for r in results[:k] if r in relevant_indices)
    return hits / len(relevant_indices) if relevant_indices else 0.0


def evaluate_dense(docs: List[str], queries: List[Dict[str, Any]]) -> Tuple[float, float]:
    searcher = DenseSearch()
    searcher.fit(docs)
    ndcgs = []
    recalls = []
    for q in queries:
        results = [i for i, _ in searcher.search(q["query"], top_k=10)]
        ndcgs.append(ndcg_at_k(results, q["relevant_doc_indices"], k=10))
        recalls.append(recall_at_k(results, q["relevant_doc_indices"], k=10))
    return sum(ndcgs) / len(ndcgs), sum(recalls) / len(recalls)


def evaluate_sparse(docs: List[str], queries: List[Dict[str, Any]]) -> Tuple[float, float]:
    searcher = SparseSearch()
    searcher.fit(docs)
    ndcgs = []
    recalls = []
    for q in queries:
        results = [i for i, _ in searcher.search(q["query"], top_k=10)]
        ndcgs.append(ndcg_at_k(results, q["relevant_doc_indices"], k=10))
        recalls.append(recall_at_k(results, q["relevant_doc_indices"], k=10))
    return sum(ndcgs) / len(ndcgs), sum(recalls) / len(recalls)


def evaluate_hybrid(docs: List[str], queries: List[Dict[str, Any]]) -> Tuple[float, float]:
    searcher = HybridSearch(dense_weight=0.7, sparse_weight=0.3)
    searcher.fit(docs)
    ndcgs = []
    recalls = []
    for q in queries:
        results = [r["id"] for r in searcher.search(q["query"])]
        ndcgs.append(ndcg_at_k(results, q["relevant_doc_indices"], k=10))
        recalls.append(recall_at_k(results, q["relevant_doc_indices"], k=10))
    return sum(ndcgs) / len(ndcgs), sum(recalls) / len(recalls)


def main() -> None:
    queries = load_queries()

    dense_ndcg, dense_recall = evaluate_dense(DOCS, queries)
    sparse_ndcg, sparse_recall = evaluate_sparse(DOCS, queries)
    hybrid_ndcg, hybrid_recall = evaluate_hybrid(DOCS, queries)

    print("=" * 70)
    print("Hybrid Retrieval Benchmark")
    print("=" * 70)
    print(f"{'Method':15s} {'NDCG@10':>10s} {'Recall@10':>10s}")
    print("-" * 70)
    print(f"{'Dense-only':15s} {dense_ndcg:>10.3f} {dense_recall:>10.3f}")
    print(f"{'Sparse-only':15s} {sparse_ndcg:>10.3f} {sparse_recall:>10.3f}")
    print(f"{'Hybrid':15s} {hybrid_ndcg:>10.3f} {hybrid_recall:>10.3f}")
    print("-" * 70)
    if dense_ndcg > 0:
        improvement = (hybrid_ndcg - dense_ndcg) / dense_ndcg
        print(f"Hybrid vs Dense NDCG improvement: {improvement:+.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
