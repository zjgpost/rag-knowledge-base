"""Build an intent-level corpus from the reply-level corpus.

For each intent, selects the top-K replies that best overlap with the
queries of that intent, and concatenates them into a single representative
document.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import re


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower()))


def build_intent_corpus(
    reply_corpus: List[Dict[str, Any]],
    queries: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    # Group reply docs by intent.
    intent_docs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for doc in reply_corpus:
        intent_docs[doc["intent"]].append(doc)

    # Group queries by intent (using the labeled relevant doc).
    intent_queries: Dict[str, List[str]] = defaultdict(list)
    for q in queries:
        # q may use 'relevant_doc_indices' (reply) or 'relevant_intents' (intent)
        if "relevant_intents" in q:
            for intent in q["relevant_intents"]:
                intent_queries[intent].append(q["query"])
        elif "relevant_doc_indices" in q:
            for idx in q["relevant_doc_indices"]:
                intent = reply_corpus[idx]["intent"]
                intent_queries[intent].append(q["query"])

    result = []
    for intent, docs in sorted(intent_docs.items()):
        query_text = " ".join(intent_queries.get(intent, []))
        query_tokens = tokens(query_text)

        # Score each doc by token overlap with queries of this intent.
        def overlap_score(doc: Dict[str, Any]) -> int:
            if not query_tokens:
                return 0
            return len(tokens(doc["content"]) & query_tokens)

        ranked = sorted(docs, key=overlap_score, reverse=True)
        selected = ranked[:top_k]

        # Build a single representative document.
        content_parts = [f"Intent: {intent}."]
        for i, doc in enumerate(selected, 1):
            content_parts.append(f"Example {i}: {doc['content']}")
        content = "\n".join(content_parts)

        result.append(
            {
                "id": len(result),
                "content": content,
                "intent": intent,
                "source_doc_ids": [d["id"] for d in selected],
            }
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build intent-level corpus from reply-level corpus"
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
        default=Path("benchmarks/dataset/ecommerce_intent_corpus.jsonl"),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of representative replies to include per intent",
    )
    args = parser.parse_args()

    reply_corpus = load_jsonl(args.reply_corpus)
    queries = load_jsonl(args.queries)

    intent_corpus = build_intent_corpus(reply_corpus, queries, top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for doc in intent_corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Built {len(intent_corpus)} intent documents from {len(reply_corpus)} replies.")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
