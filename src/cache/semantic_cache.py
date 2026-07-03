"""Semantic cache using vector similarity."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class SemanticCache:
    """Cache answers based on embedding cosine similarity.

    Uses TF-IDF as a lightweight embedding backend. In production this is
    backed by Redis Vector Search or a dedicated vector database.
    """

    def __init__(self, similarity_threshold: float = 0.92):
        self.threshold = similarity_threshold
        self.entries: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer()
        self._fitted = False

    def _ensure_fitted(self) -> None:
        if not self._fitted and self.entries:
            self.vectorizer.fit([e["query"] for e in self.entries])
            self._fitted = True

    def _embed(self, text: str) -> np.ndarray:
        if not self._fitted:
            return np.zeros((1, 1))
        return self.vectorizer.transform([text]).toarray()[0]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get(
        self,
        query: str,
        role_key: Optional[str] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]], float]:
        """Check cache for a semantically similar query under the same role."""
        if not self.entries:
            return False, None, 0.0

        self._ensure_fitted()
        query_vec = self._embed(query)

        best_match = None
        best_score = -1.0
        for entry in self.entries:
            if role_key is not None and entry.get("role_key") != role_key:
                continue
            entry_vec = self._embed(entry["query"])
            score = self._cosine_similarity(query_vec, entry_vec)
            if score > best_score:
                best_score = score
                best_match = entry

        if best_score >= self.threshold:
            return True, best_match, best_score
        return False, None, best_score

    def set(
        self,
        query: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
        role_key: Optional[str] = None,
    ) -> None:
        self.entries.append(
            {
                "query": query,
                "answer": answer,
                "metadata": metadata or {},
                "role_key": role_key,
                "hit_count": 0,
            }
        )
        # Re-fit vectorizer on all cached queries
        self.vectorizer = TfidfVectorizer()
        if self.entries:
            self.vectorizer.fit([e["query"] for e in self.entries])
            self._fitted = True

    def record_hit(self, entry: Dict[str, Any]) -> None:
        entry["hit_count"] = entry.get("hit_count", 0) + 1

    def stats(self) -> Dict[str, Any]:
        return {
            "entries": len(self.entries),
            "threshold": self.threshold,
            "total_hits": sum(e.get("hit_count", 0) for e in self.entries),
        }
