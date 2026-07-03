"""Diagnose why the real-world RAG benchmark misses relevant documents.

Outputs per-query results and failure case analysis for the best current
configuration (ST dense + BM25 sparse + cross-encoder rerank).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieval.hybrid_search import HybridSearch
from retrieval.sparse_search import SparseSearch


DATASET_DIR = Path(__file__).parent.parent / "benchmarks" / "dataset"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-st", action="store_true")
    parser.add_argument("--use-cross-encoder", action="store_true")
    parser.add_argument("--dense-weight", type=float, default=0.6)
    parser.add_argument("--sparse-weight", type=float, default=0.4)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    corpus = load_jsonl(DATASET_DIR / "ecommerce_reply_corpus.jsonl")
    queries = load_jsonl(DATASET_DIR / "ecommerce_reply_queries.jsonl")
    docs = [d["content"] for d in corpus]

    searcher = HybridSearch(
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
        dense_top_k=50,
        sparse_top_k=50,
        rerank_top_k=args.top_k,
        use_sentence_transformers=args.use_st,
        use_cross_encoder=args.use_cross_encoder,
    )
    searcher.fit(docs, metadata=corpus)

    total = len(queries)
    hits = 0
    misses = []
    partial_hits = []

    for idx, q in enumerate(queries):
        results = searcher.search(q["query"], top_k=args.top_k)
        result_ids = [r["id"] for r in results]
        relevant = set(q["relevant_doc_indices"])
        hit_positions = [i + 1 for i, rid in enumerate(result_ids) if rid in relevant]

        if hit_positions:
            hits += 1
            if min(hit_positions) > 3:
                partial_hits.append(
                    {
                        "query_idx": idx,
                        "query": q["query"],
                        "relevant": list(relevant),
                        "positions": hit_positions,
                        "top_results": result_ids[:5],
                    }
                )
        else:
            misses.append(
                {
                    "query_idx": idx,
                    "query": q["query"],
                    "relevant": list(relevant),
                    "top_results": result_ids[:5],
                }
            )

    print(f"Total queries: {total}")
    print(f"Recall@{args.top_k}: {hits}/{total} = {hits/total:.2%}")
    print(f"Complete misses: {len(misses)}")
    print(f"Low-ranked hits (>3): {len(partial_hits)}")
    print()

    print("=" * 70)
    print("SAMPLE MISSES (first 10)")
    print("=" * 70)
    for m in misses[:10]:
        print(f"\nQuery #{m['query_idx']}: {m['query']}")
        print(f"Relevant doc: {m['relevant'][0]}")
        print(f"Top 5 retrieved: {m['top_results']}")
        for rid in m["top_results"][:3]:
            doc = corpus[rid]
            print(f"  [{rid}] intent={doc['intent']} tone={doc['tone']}")
            print(f"      {doc['content'][:120]}...")

    print("\n" + "=" * 70)
    print("SAMPLE LOW-RANKED HITS (first 10)")
    print("=" * 70)
    for m in partial_hits[:10]:
        print(f"\nQuery #{m['query_idx']}: {m['query']}")
        print(f"Relevant doc: {m['relevant'][0]} at position {m['positions']}")
        print(f"Top 5 retrieved: {m['top_results']}")


if __name__ == "__main__":
    main()
