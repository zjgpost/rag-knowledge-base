"""Real-world benchmark using the SEA e-commerce customer support dataset.

Compares Dense, Sparse, and Hybrid retrieval on actual customer messages
against intent-level support documents.
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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dcg(scores: List[float]) -> float:
    return sum((2 ** s - 1) / math.log2(i + 2) for i, s in enumerate(scores))


def ndcg_at_k(results: List[int], relevant: List[int], k: int = 10) -> float:
    rel = [1 if r in relevant else 0 for r in results[:k]]
    ideal = [1] * min(len(relevant), k) + [0] * max(0, k - len(relevant))
    actual_dcg = dcg(rel)
    ideal_dcg = dcg(ideal)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(results: List[int], relevant: List[int], k: int = 10) -> float:
    hits = sum(1 for r in results[:k] if r in relevant)
    return hits / len(relevant) if relevant else 0.0


def evaluate(
    docs: List[str],
    queries: List[Dict[str, Any]],
    corpus: List[Dict[str, Any]],
    searcher_factory,
) -> Tuple[float, float]:
    searcher = searcher_factory()
    searcher.fit(docs)
    ndcgs = []
    recalls = []
    for q in queries:
        raw_results = searcher.search(q["query"], top_k=10)
        results = []
        for r in raw_results:
            if isinstance(r, dict):
                results.append(r["id"])
            elif isinstance(r, tuple):
                results.append(r[0])
            else:
                results.append(r)

        # Support both reply-level and intent-level relevance labels.
        if "relevant_doc_indices" in q:
            relevant = q["relevant_doc_indices"]
        elif "relevant_intents" in q:
            relevant = [
                i
                for i, d in enumerate(corpus)
                if d.get("intent") in q["relevant_intents"]
            ]
        else:
            relevant = []

        ndcgs.append(ndcg_at_k(results, relevant, k=10))
        recalls.append(recall_at_k(results, relevant, k=10))
    return sum(ndcgs) / len(ndcgs), sum(recalls) / len(recalls)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-world e-commerce RAG benchmark")
    parser.add_argument(
        "--use-st",
        action="store_true",
        help="Use sentence-transformers all-MiniLM-L6-v2 for dense retrieval (requires model download)",
    )
    parser.add_argument(
        "--use-cross-encoder",
        action="store_true",
        help="Use cross-encoder/ms-marco-MiniLM-L6-v2 for reranking (requires model download)",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=0.8,
        help="Hybrid dense weight (default: 0.8)",
    )
    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=0.2,
        help="Hybrid sparse weight (default: 0.2)",
    )
    parser.add_argument(
        "--dense-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model for dense retrieval (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
        help="Cross-encoder model for reranking (default: cross-encoder/ms-marco-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--use-query-expansion",
        action="store_true",
        help="Enable pseudo-relevance feedback query expansion",
    )
    parser.add_argument(
        "--expansion-terms",
        type=int,
        default=5,
        help="Number of expansion terms to append (default: 5)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DATASET_DIR / "ecommerce_reply_corpus.jsonl",
        help="Path to corpus JSONL",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DATASET_DIR / "ecommerce_reply_queries.jsonl",
        help="Path to queries JSONL",
    )
    args = parser.parse_args()

    corpus = load_jsonl(args.corpus)
    queries = load_jsonl(args.queries)

    docs = [d["content"] for d in corpus]

    dense_factory = lambda: DenseSearch(
        model_name=args.dense_model,
        use_sentence_transformers=args.use_st,
    )
    sparse_factory = lambda: SparseSearch()
    hybrid_factory = lambda: HybridSearch(
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
        use_sentence_transformers=args.use_st,
        use_cross_encoder=args.use_cross_encoder,
        dense_model_name=args.dense_model,
        reranker_model_name=args.reranker_model,
        use_query_expansion=args.use_query_expansion,
        expansion_terms=args.expansion_terms,
    )

    dense_ndcg, dense_recall = evaluate(docs, queries, corpus, dense_factory)
    sparse_ndcg, sparse_recall = evaluate(docs, queries, corpus, sparse_factory)
    hybrid_ndcg, hybrid_recall = evaluate(docs, queries, corpus, hybrid_factory)

    print("=" * 70)
    print("Real-World E-commerce RAG Benchmark")
    print(f"Corpus: {len(corpus)} documents | Queries: {len(queries)} customer messages")
    print(f"Corpus file: {args.corpus.name} | Queries file: {args.queries.name}")
    print("=" * 70)
    print(f"{'Method':15s} {'NDCG@10':>10s} {'Recall@10':>10s}")
    print("-" * 70)
    print(f"{'Dense-only':15s} {dense_ndcg:>10.3f} {dense_recall:>10.3f}")
    print(f"{'Sparse-only':15s} {sparse_ndcg:>10.3f} {sparse_recall:>10.3f}")
    print(f"{'Hybrid':15s} {hybrid_ndcg:>10.3f} {hybrid_recall:>10.3f}")
    print("-" * 70)
    if dense_ndcg > 0:
        print(f"Hybrid vs Dense NDCG improvement: {(hybrid_ndcg - dense_ndcg) / dense_ndcg:+.1%}")
        print(f"Hybrid vs Sparse NDCG improvement: {(hybrid_ndcg - sparse_ndcg) / sparse_ndcg:+.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
