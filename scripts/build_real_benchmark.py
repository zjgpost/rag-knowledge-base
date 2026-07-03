"""Analyze dataset intents and build a real benchmark."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset


def main():
    ds = load_dataset("nwchang/sea-ecommerce-customer-support-sample")
    train = ds["train"]

    intents = Counter(train["intent"])
    print(f"Total examples: {len(train)}")
    print(f"Intents ({len(intents)}):")
    for intent, count in intents.most_common():
        print(f"  {intent}: {count}")

    # Group replies by intent to create document corpus
    docs_by_intent: dict[str, list[str]] = {}
    queries_by_intent: dict[str, list[str]] = {}
    for row in train:
        intent = row["intent"]
        docs_by_intent.setdefault(intent, []).append(row["assistant_reply"])
        queries_by_intent.setdefault(intent, []).append(row["customer_message"])

    # Save corpus and queries
    out_dir = Path(__file__).parent.parent / "benchmarks" / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = []
    for intent, replies in docs_by_intent.items():
        # Aggregate up to 20 replies per intent to keep docs reasonable
        selected = replies[:20]
        content = "\n".join(selected)
        corpus.append({"id": intent, "content": content, "intent": intent})

    with open(out_dir / "ecommerce_corpus.jsonl", "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Sample 5 queries per intent for benchmark
    queries = []
    for intent, msgs in queries_by_intent.items():
        for msg in msgs[:5]:
            queries.append({"query": msg, "relevant_intents": [intent]})

    with open(out_dir / "ecommerce_queries.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(corpus)} intent documents and {len(queries)} queries.")


if __name__ == "__main__":
    main()
