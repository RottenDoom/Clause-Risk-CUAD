"""
services/retrieval/retriever_chroma.py

ChromaDB query interface — original implementation, kept for reference.
The active retriever is services/retrieval/retriever.py (Pinecone).

To switch back: replace retriever.py with this file's content.
"""

from typing import Any

from config import CHROMA_DB_PATH, COLLECTION_NAMES, PRECEDENT_CONTRAST_FETCH_K, PRECEDENT_SIMILAR_TOP_K


class Retriever:
    """
    Thin wrapper around ChromaDB that exposes the two query patterns
    needed by the precedent retrieval step.
    """

    def __init__(self, db_path: str = CHROMA_DB_PATH) -> None:
        import chromadb
        self._client = chromadb.PersistentClient(path=db_path)

    def query(
        self,
        embedding: list[float],
        family: str,
        n_results: int = PRECEDENT_SIMILAR_TOP_K,
    ) -> list[dict[str, Any]]:
        """
        Query the family collection for the nearest neighbours.

        Returns a list of dicts:
          {"contract_id": str, "chunk_text": str, "full_clause_text": str, "distance": float}

        Over-fetches by 3x to allow the caller to deduplicate by contract_id.
        """
        collection_name = COLLECTION_NAMES[family]
        try:
            collection = self._client.get_collection(collection_name)
        except Exception:
            return []

        count = collection.count()
        if count == 0:
            return []

        fetch = min(n_results * 3, count)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=fetch,
            include=["documents", "metadatas", "distances"],
        )

        output: list[dict[str, Any]] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "contract_id": meta["contract_id"],
                "chunk_text": doc,
                "full_clause_text": meta.get("full_clause_text", doc),
                "distance": dist,
            })
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
        """
        collection_name = COLLECTION_NAMES[family]
        try:
            collection = self._client.get_collection(collection_name)
        except Exception:
            return []

        count = collection.count()
        if count == 0:
            return []

        fetch = min(n_results, count)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=fetch,
            include=["documents", "metadatas"],
        )

        output: list[dict[str, Any]] = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            output.append({
                "contract_id": meta["contract_id"],
                "chunk_text": doc,
                "full_clause_text": meta.get("full_clause_text", doc),
                "distance": None,
            })
        return output

    def collection_count(self, family: str) -> int:
        """Return the number of documents in the family collection, or 0 if missing."""
        try:
            return self._client.get_collection(COLLECTION_NAMES[family]).count()
        except Exception:
            return 0
