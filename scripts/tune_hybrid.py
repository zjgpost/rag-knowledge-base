"""Quick grid search for BM25 + hybrid weights on the real-world benchmark.

Caches dense vectors so the grid search finishes in seconds instead of
re-encoding the corpus for every weight combination.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieval.dense_search import DenseSearch
from retrieval.sparse_search import SparseSearch


DATASET_DIR = Path(__file__).parent.parent / "benchmarks" / "dataset"


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


def encode_dense(
    docs: List[str], use_st: bool, model_name: str
) -> Tuple[np.ndarray, Any]:
    """Return document vectors and the query encoder (vectorizer or model)."""
    ds = DenseSearch(model_name=model_name, use_sentence_transformers=use_st)
    ds.fit(docs)
    return ds.vectors, ds


def dense_scores_for_query(
    query: str, encoder: Any
) -> Tuple[List[int], np.ndarray]:
    """Return all doc indices and cosine scores for a query."""
    ds = encoder
    query = ds._normalize(query)
    if ds._use_st:
        query_vec = ds._model.encode([query], show_progress_bar=False)
    else:
        q_tfidf = ds._vectorizer.transform([query]).toarray()
        if ds._svd is not None:
            query_vec = ds._svd.transform(q_tfidf)
        else:
            query_vec = q_tfidf

    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    doc_vecs = ds.vectors / (np.linalg.norm(ds.vectors, axis=1, keepdims=True) + 1e-10)
    scores = np.dot(doc_vecs, query_vec.T).flatten()
    return list(range(len(scores))), scores


def evaluate_grid(
    docs: List[str],
    queries: List[Dict[str, Any]],
    corpus: List[Dict[str, Any]],
    dense_encoder: Any,
    sparse_searcher: SparseSearch,
    dense_weight: float,
    sparse_weight: float,
) -> Tuple[float, float]:
    ndcgs = []
    recalls = []
    for q in queries:
        dense_ids, dense_scores = dense_scores_for_query(q["query"], dense_encoder)
        sparse_results = sparse_searcher.search(q["query"], top_k=50)

        fused: Dict[int, float] = {}
        max_dense = float(np.max(dense_scores)) if len(dense_scores) else 0.0
        if max_dense > 0:
            for doc_id, score in zip(dense_ids, dense_scores):
                fused[doc_id] = dense_weight * (score / max_dense)

        if sparse_results:
            max_sparse = max(score for _, score in sparse_results)
            for doc_id, score in sparse_results:
                if max_sparse > 0:
                    fused[doc_id] = fused.get(doc_id, 0.0) + sparse_weight * (score / max_sparse)

        results = sorted(fused, key=fused.get, reverse=True)[:10]

        if "relevant_doc_indices" in q:
            relevant = q["relevant_doc_indices"]
        elif "relevant_intents" in q:
            relevant = [
                i for i, d in enumerate(corpus) if d.get("intent") in q["relevant_intents"]
            ]
        else:
            relevant = []

        ndcgs.append(ndcg_at_k(results, relevant, k=10))
        recalls.append(recall_at_k(results, relevant, k=10))

    return sum(ndcgs) / len(ndcgs), sum(recalls) / len(recalls)


def main():
    parser = argparse.ArgumentParser(description="Grid-search hybrid weights")
    parser.add_argument(
        "--use-st",
        action="store_true",
        help="Use sentence-transformers for dense retrieval",
    )
    parser.add_argument(
        "--dense-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model for dense retrieval",
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

    print("Encoding dense vectors once...")
    dense_vectors, dense_encoder = encode_dense(
        docs, use_st=args.use_st, model_name=args.dense_model
    )
    print(f"Dense vectors shape: {dense_vectors.shape}")

    best = {"ndcg": 0.0}
    for k1 in [1.0, 1.2, 1.5, 1.8, 2.0]:
        for b in [0.5, 0.75, 0.9]:
            sparse = SparseSearch(k1=k1, b=b)
            sparse.fit(docs)
            for dense_w in [0.6, 0.7, 0.8, 0.9]:
                sparse_w = 1 - dense_w
                ndcg, recall = evaluate_grid(
                    docs, queries, corpus, dense_encoder, sparse, dense_w, sparse_w
                )
                if ndcg > best["ndcg"]:
                    best.update(
                        {"ndcg": ndcg, "recall": recall, "k1": k1, "b": b, "dense_w": dense_w}
                    )
                print(f"k1={k1} b={b} dw={dense_w} => NDCG={ndcg:.3f} Recall={recall:.3f}")

    print("\nBest:", best)


if __name__ == "__main__":
    main()
