"""
agent/embedder.py

Thin singleton wrapper around sentence-transformers so the model is loaded
once per process and reused across clause_discovery and build_index calls.
Embeddings are L2-normalised, making dot product == cosine similarity.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from config import EMBEDDING_MODEL

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            sys.exit("ERROR: sentence-transformers not installed. Run: uv pip install sentence-transformers")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed(texts: list[str], show_progress: bool = False) -> np.ndarray:
    """
    Embed a batch of texts. Returns shape (len(texts), EMBEDDING_DIM), float32,
    L2-normalised so dot(a, b) == cosine_similarity(a, b).
    """
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=show_progress)


def embed_one(text: str) -> np.ndarray:
    """Embed a single string. Returns shape (EMBEDDING_DIM,)."""
    return embed([text])[0]

def cosine_similarity_matrix(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Computes pairwise cosine similarity between query_vecs (shape M x D)
    and doc_vecs (shape N x D). Returns a matrix of shape (M, N).
    Since normalize_embeddings=True in embed_texts, this reduces to a dot product.
    """
    return np.dot(query_vecs, doc_vecs.T)
