"""Dense vector retrieval using embeddings.

Tries to use sentence-transformers (all-MiniLM-L6-v2) if explicitly enabled.
Falls back to TF-IDF, or to TruncatedSVD on TF-IDF for a lightweight dense vector.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


_PLACEHOLDER_RE = re.compile(r"\[([A-Z_]+)\]")

# Local model cache inside the repo (populated by scripts/download_models.py).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _local_model_dir(model_name: str) -> Path:
    """Map a HuggingFace model id to a local directory name."""
    safe_name = model_name.replace("/", "--").replace(" ", "_")
    return _PROJECT_ROOT / "models" / safe_name


class DenseSearch:
    """Dense retrieval with sentence-transformers or TF-IDF/SVD fallback."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_sentence_transformers: bool = False,
        use_svd: bool = False,
        svd_components: int = 128,
    ):
        self.model_name = model_name
        self.use_sentence_transformers = use_sentence_transformers
        self.use_svd = use_svd
        self.svd_components = svd_components
        self.documents: List[str] = []
        self.vectors: np.ndarray | None = None
        self._model = None
        self._vectorizer = None
        self._svd = None
        self._use_st = False

    @staticmethod
    def _normalize(text: str) -> str:
        # Make placeholders like [ORDER_ID] searchable as ORDER_ID.
        return _PLACEHOLDER_RE.sub(r" \1 ", text)

    def _load_model(self):
        if self._model is not None or self._vectorizer is not None:
            return
        if self.use_sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer

                # Prefer locally downloaded model so the repo works offline.
                local_dir = _local_model_dir(self.model_name)
                if local_dir.exists():
                    model_path = str(local_dir)
                else:
                    # Some download scripts drop the org prefix (e.g.
                    # sentence-transformers/... -> models/paraphrase-...).
                    short_dir = _PROJECT_ROOT / "models" / Path(self.model_name).name
                    model_path = str(short_dir) if short_dir.exists() else self.model_name
                self._model = SentenceTransformer(model_path)
                self._use_st = True
                return
            except Exception:
                pass
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
            token_pattern=r"(?u)\b\w[\w']*\b",
        )
        self._use_st = False

    def fit(self, documents: List[str]) -> None:
        self._load_model()
        self.documents = [self._normalize(d) for d in documents]
        if self._use_st:
            self.vectors = self._model.encode(self.documents, show_progress_bar=False)
        else:
            tfidf = self._vectorizer.fit_transform(self.documents).toarray()
            if self.use_svd and min(tfidf.shape) > self.svd_components:
                from sklearn.decomposition import TruncatedSVD

                self._svd = TruncatedSVD(n_components=self.svd_components)
                self.vectors = self._svd.fit_transform(tfidf)
            else:
                self.vectors = tfidf

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.vectors is None or not self.documents:
            return []

        query = self._normalize(query)
        if self._use_st:
            query_vec = self._model.encode([query], show_progress_bar=False)
        else:
            q_tfidf = self._vectorizer.transform([query]).toarray()
            if self._svd is not None:
                query_vec = self._svd.transform(q_tfidf)
            else:
                query_vec = q_tfidf

        # Normalize for cosine similarity
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        doc_vecs = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10)
        scores = np.dot(doc_vecs, query_vec.T).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]
