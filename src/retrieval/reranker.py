"""Reranker with optional cross-encoder support.

Production systems often use Cohere Rank API or a cross-encoder.
This module provides:
- a lightweight keyword-overlap heuristic that needs no model download
- an optional cross-encoder reranker (ms-marco-MiniLM) when available locally
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


_PLACEHOLDER_RE = re.compile(r"\[([A-Z_]+)\]")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _local_model_dir(model_name: str) -> Path:
    """Map a HuggingFace model id to a local directory name."""
    safe_name = model_name.replace("/", "--").replace(" ", "_")
    return _PROJECT_ROOT / "models" / safe_name

class Reranker:
    """Reranker with keyword overlap fallback and optional cross-encoder."""

    def __init__(
        self,
        use_cross_encoder: bool = False,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
    ):
        self.use_cross_encoder = use_cross_encoder
        self.model_name = model_name
        self._model = None

    @staticmethod
    def _normalize(text: str) -> str:
        return _PLACEHOLDER_RE.sub(r" \1 ", text)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(
            re.findall(r"\b[a-zA-Z0-9_]+\b", Reranker._normalize(text).lower())
        )

    def _load_cross_encoder(self):
        if self._model is not None or not self.use_cross_encoder:
            return
        try:
            from sentence_transformers import CrossEncoder

            local_dir = _local_model_dir(self.model_name)
            if local_dir.exists():
                model_path = str(local_dir)
            else:
                # Some download scripts drop the org prefix.
                short_dir = _PROJECT_ROOT / "models" / Path(self.model_name).name
                model_path = str(short_dir) if short_dir.exists() else self.model_name
            self._model = CrossEncoder(model_path, max_length=512)
        except Exception:
            self._model = None

    @staticmethod
    def _overlap_score(query: str, doc: str) -> float:
        query_tokens = Reranker._tokens(query)
        doc_tokens = Reranker._tokens(doc)
        if not doc_tokens:
            return 0.0
        overlap = len(query_tokens & doc_tokens)
        coverage = overlap / max(len(query_tokens), 1)
        density = overlap / max(len(doc_tokens), 1)
        return 0.7 * coverage + 0.3 * density

    def rank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
        base_scores: Optional[Dict[int, float]] = None,
    ) -> List[Tuple[int, float]]:
        self._load_cross_encoder()

        if self._model is not None:
            pairs = [(query, doc) for doc in documents]
            logits = self._model.predict(pairs, show_progress_bar=False)
            ce_scores = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float)))
        else:
            ce_scores = None

        scores = []
        doc_ids = list(base_scores.keys()) if base_scores else None
        for idx, doc in enumerate(documents):
            if ce_scores is not None:
                score = float(ce_scores[idx])
                if base_scores and doc_ids is not None:
                    base = base_scores.get(doc_ids[idx], 0.0)
                    score = 0.7 * score + 0.3 * base
            else:
                score = self._overlap_score(query, doc)
                if base_scores and doc_ids is not None:
                    base = base_scores.get(doc_ids[idx], 0.0)
                    score = 0.75 * base + 0.25 * score
            scores.append(score)

        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in indexed[:top_k]]
