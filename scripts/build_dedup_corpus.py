"""Build a deduplicated reply corpus with one representative per intent+tone.

This reduces the 1000 reply documents down to ~60-80 representative documents
(one per intent+tone combination), keeping the reply that overlaps most with
queries of that intent.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower()))


def build_deduplicated_corpus(
    reply_corpus: List[Dict[str, Any]],
    queries: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[int, int]]:
    # Collect query tokens per intent.
    intent_query_tokens: Dict[str, set[str]] = defaultdict(set)
    for q in queries:
        for idx in q.get("relevant_doc_indices", []):
            intent = reply_corpus[idx]["intent"]
            intent_query_tokens[intent] |= tokens(q["query"])

    # Group docs by (intent, tone).
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for doc in reply_corpus:
        grouped[(doc["intent"], doc["tone"])].append(doc)

    result = []
    old_to_new: Dict[int, int] = {}
    for (intent, tone), docs in sorted(grouped.items()):
        q_tokens = intent_query_tokens.get(intent, set())

        def score(doc: Dict[str, Any]) -> int:
            if not q_tokens:
                return 0
            return len(tokens(doc["content"]) & q_tokens)

        best = max(docs, key=score)
        old_to_new[best["id"]] = len(result)
        # Map any doc of this (intent, tone) to the selected representative.
        for d in docs:
            old_to_new[d["id"]] = len(result)

        result.append(
            {
                "id": len(result),
                "content": best["content"],
                "intent": intent,
                "tone": tone,
                "source_doc_id": best["id"],
            }
        )

    return result, old_to_new


def main():
    parser = argparse.ArgumentParser(
        description="Build deduplicated intent+tone corpus from reply corpus"
    )
    parser.add_argument(
        "--reply-corpus",
        type=Path,
        default=Path("benchmarks/dataset/ecommerce_reply_corpus.jsonl"),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("benchmarks/dataset/ecommerce_reply_queries.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/dataset/ecommerce_dedup_corpus.jsonl"),
    )
    parser.add_argument(
        "--output-queries",
        type=Path,
        default=Path("benchmarks/dataset/ecommerce_dedup_queries.jsonl"),
    )
    args = parser.parse_args()

    reply_corpus = load_jsonl(args.reply_corpus)
    queries = load_jsonl(args.queries)

    dedup_corpus, old_to_new = build_deduplicated_corpus(reply_corpus, queries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for doc in dedup_corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    remapped_queries = []
    for q in queries:
        new_indices = [old_to_new[idx] for idx in q["relevant_doc_indices"]]
        remapped_queries.append(
            {
                **q,
                "relevant_doc_indices": new_indices,
            }
        )
    with open(args.output_queries, "w", encoding="utf-8") as f:
        for q in remapped_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Built {len(dedup_corpus)} deduplicated documents from {len(reply_corpus)} replies.")
    print(f"Saved corpus to {args.output}")
    print(f"Saved remapped queries to {args.output_queries}")


if __name__ == "__main__":
    main()
