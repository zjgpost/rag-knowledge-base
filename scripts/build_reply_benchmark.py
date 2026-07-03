"""Build a harder real-world benchmark: retrieve the exact reply for each query."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset


def main():
    ds = load_dataset("nwchang/sea-ecommerce-customer-support-sample")
    train = ds["train"]

    out_dir = Path(__file__).parent.parent / "benchmarks" / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use every assistant_reply as a document, enriched with intent/tone metadata
    # so keyword (BM25) retrievers can match on category as well as content.
    corpus = []
    for i, row in enumerate(train):
        content = (
            f"Intent: {row['intent']}. "
            f"Tone: {row['tone']}. "
            f"{row['assistant_reply']}"
        )
        corpus.append(
            {
                "id": i,
                "content": content,
                "intent": row["intent"],
                "tone": row["tone"],
            }
        )

    with open(out_dir / "ecommerce_reply_corpus.jsonl", "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Sample 100 queries, each relevant to its own reply index
    queries = []
    step = max(1, len(train) // 100)
    for i in range(0, len(train), step):
        if len(queries) >= 100:
            break
        row = train[i]
        queries.append(
            {
                "query": row["customer_message"],
                "relevant_doc_indices": [i],
            }
        )

    with open(out_dir / "ecommerce_reply_queries.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Saved {len(corpus)} reply documents and {len(queries)} queries.")


if __name__ == "__main__":
    main()
