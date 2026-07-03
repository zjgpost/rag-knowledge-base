"""Basic RAG example using the knowledge base."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kb import KnowledgeBase


def main() -> None:
    docs = [
        "Redis connection pool max_connections controls the maximum number of client connections. "
        "When the pool is exhausted, new connections will be rejected. Increase max_connections or "
        "use connection pooling in your application to reuse connections.",
        "Redis performance optimization involves tuning memory policies, persistence settings, and "
        "network parameters. Use INFO stats to identify bottlenecks.",
        "Linux server monitoring uses tools like top, htop, vmstat, and iostat to observe CPU, memory, "
        "disk, and network utilization.",
    ]

    metadata = [
        {"department": "engineering", "access_level": "internal"},
        {"department": "engineering", "access_level": "internal"},
        {"department": "operations", "access_level": "public"},
    ]

    kb = KnowledgeBase()
    kb.add_documents(docs, metadata=metadata, chunk_strategy="semantic")

    # First query populates cache
    result1 = kb.query(
        "Redis connection pool full",
        user_role={"department": "engineering", "clearance": "internal"},
    )
    print("Q1:", result1["question"])
    print("Source:", result1["source"])
    print("Answer:", result1["answer"][:200])
    print()

    # Semantically similar query hits cache
    result2 = kb.query(
        "Redis connection pool is full",
        user_role={"department": "engineering", "clearance": "internal"},
    )
    print("Q2:", result2["question"])
    print("Source:", result2["source"])
    print("Answer:", result2["answer"][:200])
    print()

    # RBAC: operations user cannot see engineering docs
    result3 = kb.query("Redis connection pool full", user_role={"department": "operations"})
    print("Q3 (RBAC filtered):", result3["question"])
    print("Answer:", result3["answer"][:200])
    print()

    print("Cache stats:", kb.cache_stats())


if __name__ == "__main__":
    main()
