"""
agent/embedder.py

Thin singleton wrapper around sentence-transformers so the model is loaded
once per process and reused across clause_discovery and build_index calls.
Embeddings are L2-normalised, making dot product == cosine similarity.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from config import EMBEDDING_MODEL, MODELS_CACHE_DIR

# Force offline mode so HuggingFace doesn't try to phone home for update checks.
# If the model files exist locally, load is instant; otherwise the user must
# download the model first (one-time, via build_index.py or by deleting the
# HF_HUB_OFFLINE flag).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(MODELS_CACHE_DIR))

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # double-checked locking
                MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "Loading embedding model %s (cache=%s)…",
                    EMBEDDING_MODEL, MODELS_CACHE_DIR,
                )
                t0 = time.monotonic()
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError:
                    sys.exit(
                        "ERROR: sentence-transformers not installed. "
                        "Run: uv pip install sentence-transformers"
                    )
                _model = SentenceTransformer(
                    EMBEDDING_MODEL, cache_folder=str(MODELS_CACHE_DIR)
                )
                logger.info("Embedding model ready (%.1fs)", time.monotonic() - t0)
    return _model


def embed(texts: list[str], show_progress: bool = False) -> np.ndarray:
    """
    Embed a batch of texts. Returns shape (len(texts), EMBEDDING_DIM), float32,
    L2-normalised so dot(a, b) == cosine_similarity(a, b).
    """
    tid = threading.get_ident()
    n = len(texts)
    t0 = time.monotonic()
    logger.info("embed[t%d] start n=%d", tid, n)
    model = _get_model()
    t1 = time.monotonic()
    out = model.encode(texts, normalize_embeddings=True, show_progress_bar=show_progress)
    logger.info(
        "embed[t%d] done n=%d shape=%s model_get=%.2fs encode=%.2fs total=%.2fs",
        tid, n, out.shape, t1 - t0, time.monotonic() - t1, time.monotonic() - t0,
    )
    return out


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
