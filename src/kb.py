"""KnowledgeBase main entry point."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from cache.semantic_cache import SemanticCache
from ingestion.chunker import chunk_text
from rbac.metadata_filter import MetadataFilter
from retrieval.hybrid_search import HybridSearch


class KnowledgeBase:
    """End-to-end RAG knowledge base with hybrid retrieval and semantic cache."""

    def __init__(
        self,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        cache_threshold: float = 0.92,
    ):
        self.searcher = HybridSearch(
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        self.cache = SemanticCache(similarity_threshold=cache_threshold)
        self._cache_hits = 0
        self._cache_misses = 0

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        chunk_strategy: str = "semantic",
    ) -> None:
        """Chunk and index documents."""
        metadata = metadata or [{} for _ in documents]
        all_chunks = []
        all_meta = []
        for doc, meta in zip(documents, metadata):
            chunks = chunk_text(doc, strategy=chunk_strategy)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_meta.append(meta.copy())
        self.searcher.fit(all_chunks, all_meta)

    def query(
        self,
        question: str,
        user_role: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Query the knowledge base."""
        role_key = str(sorted(user_role.items())) if user_role else None

        if use_cache:
            hit, cached, score = self.cache.get(question, role_key=role_key)
            if hit:
                self.cache.record_hit(cached)
                self._cache_hits += 1
                return {
                    "question": question,
                    "answer": cached["answer"],
                    "source": "cache",
                    "similarity": score,
                    "documents": [],
                }
            self._cache_misses += 1

        # RBAC filtering
        allowed_ids = None
        if user_role is not None:
            filter_obj = MetadataFilter(user_role)
            allowed_ids = filter_obj.filter_doc_ids(self.searcher.doc_metadata)

        documents = self.searcher.search(question, allowed_doc_ids=allowed_ids)

        # Simple answer synthesis: concatenate top documents
        answer = self._synthesize_answer(question, documents)

        if use_cache:
            self.cache.set(question, answer, role_key=role_key)

        return {
            "question": question,
            "answer": answer,
            "source": "retrieval",
            "documents": documents,
        }

    @staticmethod
    def _synthesize_answer(question: str, documents: List[Dict[str, Any]]) -> str:
        if not documents:
            return "I could not find relevant information."
        context = "\n".join(d["content"] for d in documents[:3])
        return f"Based on the retrieved documents:\n{context}"

    def cache_stats(self) -> Dict[str, Any]:
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total > 0 else 0.0,
            **self.cache.stats(),
        }
