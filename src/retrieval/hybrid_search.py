"""Hybrid search: Dense + Sparse + Rerank."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from retrieval.dense_search import DenseSearch
from retrieval.query_expansion import PRFQueryExpander
from retrieval.reranker import Reranker
from retrieval.sparse_search import SparseSearch


class HybridSearch:
    """Three-stage retrieval: Dense recall, Sparse recall, Rerank fusion."""

    def __init__(
        self,
        dense_weight: float = 0.8,
        sparse_weight: float = 0.2,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        rerank_top_k: int = 10,
        use_sentence_transformers: bool = False,
        use_cross_encoder: bool = False,
        dense_model_name: str = "all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        use_query_expansion: bool = False,
        expansion_top_k: int = 3,
        expansion_terms: int = 5,
    ):
        self.dense = DenseSearch(
            model_name=dense_model_name,
            use_sentence_transformers=use_sentence_transformers,
        )
        self.sparse = SparseSearch()
        self.reranker = Reranker(
            use_cross_encoder=use_cross_encoder,
            model_name=reranker_model_name,
        )
        self.use_query_expansion = use_query_expansion
        self.expansion_top_k = expansion_top_k
        self.expansion_terms = expansion_terms
        self._query_expander: PRFQueryExpander | None = None
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rerank_top_k = rerank_top_k
        self.documents: List[str] = []
        self.doc_metadata: List[Dict[str, Any]] = []

    def fit(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.documents = documents
        self.doc_metadata = metadata or [{} for _ in documents]
        self.dense.fit(documents)
        self.sparse.fit(documents)
        if self.use_query_expansion:
            self._query_expander = PRFQueryExpander(
                self.sparse, top_k=self.expansion_top_k, num_terms=self.expansion_terms
            )

    def _score_fusion(
        self,
        dense_results: List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]],
    ) -> Dict[int, float]:
        """Normalize scores to [0, 1] and compute weighted sum."""
        scores: Dict[int, float] = {}

        if dense_results:
            max_score = max(score for _, score in dense_results)
            for doc_id, score in dense_results:
                norm = score / max_score if max_score > 0 else 0
                scores[doc_id] = scores.get(doc_id, 0) + self.dense_weight * norm

        if sparse_results:
            max_score = max(score for _, score in sparse_results)
            for doc_id, score in sparse_results:
                norm = score / max_score if max_score > 0 else 0
                scores[doc_id] = scores.get(doc_id, 0) + self.sparse_weight * norm

        return scores

    def search(
        self,
        query: str,
        allowed_doc_ids: Optional[List[int]] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run hybrid search and return reranked documents."""
        top_k = top_k or self.rerank_top_k
        if not self.documents:
            return []

        if self.use_query_expansion and self._query_expander is not None:
            query = self._query_expander.expand(query)

        dense_results = self.dense.search(query, top_k=self.dense_top_k)
        sparse_results = self.sparse.search(query, top_k=self.sparse_top_k)

        fused = self._score_fusion(dense_results, sparse_results)
        if allowed_doc_ids is not None:
            fused = {k: v for k, v in fused.items() if k in allowed_doc_ids}

        if not fused:
            return []

        candidate_ids = sorted(fused, key=fused.get, reverse=True)
        candidate_docs = [self.documents[i] for i in candidate_ids]
        reranked = self.reranker.rank(
            query, candidate_docs, top_k=top_k, base_scores=fused
        )

        results = []
        for local_idx, score in reranked:
            doc_id = candidate_ids[local_idx]
            results.append(
                {
                    "id": doc_id,
                    "content": self.documents[doc_id],
                    "metadata": self.doc_metadata[doc_id],
                    "hybrid_score": fused[doc_id],
                    "rerank_score": score,
                }
            )
        return results

    def set_weights(self, dense_weight: float, sparse_weight: float) -> None:
        total = dense_weight + sparse_weight
        self.dense_weight = dense_weight / total
        self.sparse_weight = sparse_weight / total
