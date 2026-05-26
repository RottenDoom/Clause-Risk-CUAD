"""
services/retrieval/retriever.py

Pinecone query interface for precedent retrieval.

Architecture:
  - One serverless index (PINECONE_INDEX_NAME = "cuad-contracts")
  - Four namespaces, one per clause family (assignment, change_of_control, ...)
  - Cosine metric; dimension 384 (all-MiniLM-L6-v2)

Return format for all query methods is identical to the ChromaDB version so
that agent/precedent_retrieval.py requires no changes.

Distance convention: Pinecone returns similarity scores (higher = closer).
We expose distance = 1 - score to match the ChromaDB convention used upstream.

ChromaDB backup: services/retrieval/retriever_chroma.py
"""

import logging
import os
import time
from typing import Any

from config import (
    PINECONE_INDEX_NAME,
    PRECEDENT_CONTRAST_FETCH_K,
    PRECEDENT_SIMILAR_TOP_K,
)

logger = logging.getLogger(__name__)

# Pinecone metadata fields that hold the text — stored at upsert time
# because Pinecone has no separate document store.
_CHUNK_TEXT_KEY = "chunk_text"
_FULL_CLAUSE_KEY = "full_clause_text"


class Retriever:
    """
    Thin wrapper around a Pinecone index that exposes the two query patterns
    needed by the precedent retrieval step.

    Requires PINECONE_API_KEY in the environment (loaded from .env by config).
    """

    def __init__(self) -> None:
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "PINECONE_API_KEY is not set. Add it to your .env file."
            )
        from pinecone import Pinecone
        pc = Pinecone(api_key=api_key)
        self._index = pc.Index(PINECONE_INDEX_NAME)

    def query(
        self,
        embedding: list[float],
        family: str,
        n_results: int = PRECEDENT_SIMILAR_TOP_K,
    ) -> list[dict[str, Any]]:
        """
        Query the family namespace for the nearest neighbours.

        Returns a list of dicts:
          {"contract_id": str, "chunk_text": str, "full_clause_text": str, "distance": float}

        Over-fetches by 3x to allow the caller to deduplicate by contract_id.
        """
        fetch = min(n_results * 3, 100)
        t0 = time.monotonic()
        try:
            response = self._index.query(
                vector=embedding,
                top_k=fetch,
                namespace=family,
                include_metadata=True,
            )
        except Exception as exc:
            logger.error("Pinecone query failed family=%s — %s", family, exc)
            return []

        ms = int((time.monotonic() - t0) * 1000)
        output: list[dict[str, Any]] = []
        for match in response.matches:
            meta = match.metadata or {}
            output.append({
                "contract_id": meta.get("contract_id", match.id.split("__")[0]),
                "chunk_text": meta.get(_CHUNK_TEXT_KEY, ""),
                "full_clause_text": meta.get(_FULL_CLAUSE_KEY, ""),
                "distance": 1.0 - match.score,
            })
        logger.debug("query family=%s fetched=%d returned=%d dur=%dms", family, fetch, len(output), ms)
        return output

    def query_for_contrast(
        self,
        embedding: list[float],
        family: str,
        n_results: int = PRECEDENT_CONTRAST_FETCH_K,
    ) -> list[dict[str, Any]]:
        """
        Over-fetch candidates for contrasting precedent selection.
        Same return format as query(); caller deduplicates and passes to LLM for selection.
        distance is None — not needed for contrast selection.
        """
        fetch = min(n_results, 100)
        t0 = time.monotonic()
        try:
            response = self._index.query(
                vector=embedding,
                top_k=fetch,
                namespace=family,
                include_metadata=True,
            )
        except Exception as exc:
            logger.error("Pinecone contrast query failed family=%s — %s", family, exc)
            return []

        ms = int((time.monotonic() - t0) * 1000)
        output: list[dict[str, Any]] = []
        for match in response.matches:
            meta = match.metadata or {}
            output.append({
                "contract_id": meta.get("contract_id", match.id.split("__")[0]),
                "chunk_text": meta.get(_CHUNK_TEXT_KEY, ""),
                "full_clause_text": meta.get(_FULL_CLAUSE_KEY, ""),
                "distance": None,
            })
        logger.debug("query_for_contrast family=%s fetched=%d returned=%d dur=%dms", family, fetch, len(output), ms)
        return output

    def collection_count(self, family: str) -> int:
        """Return the number of vectors in the family namespace, or 0 on any error."""
        try:
            stats = self._index.describe_index_stats()
            ns_summary = stats.namespaces.get(family)
            return ns_summary.vector_count if ns_summary else 0
        except Exception:
            return 0
