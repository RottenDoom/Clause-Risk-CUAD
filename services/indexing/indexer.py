"""
services/indexing/indexer.py

Pinecone indexing service.

Reads reference_annotations.json, chunks each clause, embeds with
sentence-transformers, and upserts into one Pinecone serverless index
using four namespaces — one per clause family.

chunk_text is stored in the metadata dict (Pinecone has no separate document
store). full_clause_text is capped at METADATA_TEXT_LIMIT to stay well under
Pinecone's 40 KB per-vector metadata limit.

ChromaDB backup: services/indexing/indexer_chroma.py
"""

import json
import os
from pathlib import Path

from config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    CLAUSE_FAMILIES,
    COLLECTION_NAMES,
    EMBEDDING_DIM,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
)

_BATCH_SIZE = 100
_METADATA_TEXT_LIMIT = 20_000  # 20 KB — well under the 40 KB Pinecone limit


class Indexer:
    """
    Offline indexing service for Pinecone.

    Requires PINECONE_API_KEY in the environment.
    Re-running is safe — upsert is idempotent.
    """

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.embedding_model = embedding_model

    def build(self, annotations_path: Path, reference_dir: Path) -> dict[str, int]:  # noqa: ARG002
        """
        Build the full Pinecone index from reference annotations.
        Returns {family: vector_count} upserted per namespace.
        """
        from agent.embedder import embed

        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not set.")

        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=api_key)
        self._ensure_index(pc)
        index = pc.Index(PINECONE_INDEX_NAME)

        with open(annotations_path, encoding="utf-8") as f:
            annotations: dict[str, dict[str, str | None]] = json.load(f)

        pending: list[dict] = []
        for contract_id, family_texts in annotations.items():
            for family in CLAUSE_FAMILIES:
                clause_text = family_texts.get(family)
                if not clause_text or not clause_text.strip():
                    continue
                for chunk_idx, (_, __, chunk_str) in enumerate(self._chunk(clause_text)):
                    pending.append({
                        "family": family,
                        "contract_id": contract_id,
                        "chunk_idx": chunk_idx,
                        "text": chunk_str,
                        "full_clause_text": clause_text,
                    })

        if not pending:
            return {f: 0 for f in CLAUSE_FAMILIES}

        all_embeddings = embed([p["text"] for p in pending], show_progress=True)

        by_family: dict[str, list[tuple[int, dict]]] = {f: [] for f in CLAUSE_FAMILIES}
        for i, record in enumerate(pending):
            by_family[record["family"]].append((i, record))

        counts: dict[str, int] = {}
        for family in CLAUSE_FAMILIES:
            items = by_family[family]
            if not items:
                counts[family] = 0
                continue

            seen: dict[str, int] = {}
            vectors = []
            for global_idx, record in items:
                cid = record["contract_id"]
                chunk_idx = seen.get(cid, 0)
                seen[cid] = chunk_idx + 1
                vectors.append({
                    "id": f"{cid}__{family}__{chunk_idx}",
                    "values": all_embeddings[global_idx].tolist(),
                    "metadata": {
                        "contract_id": cid,
                        "clause_family": family,
                        "chunk_text": record["text"][:_METADATA_TEXT_LIMIT],
                        "full_clause_text": record["full_clause_text"][:_METADATA_TEXT_LIMIT],
                        "chunk_index": chunk_idx,
                    },
                })

            for i in range(0, len(vectors), _BATCH_SIZE):
                index.upsert(vectors=vectors[i:i + _BATCH_SIZE], namespace=family)

            counts[family] = len(vectors)

        return counts

    @staticmethod
    def _ensure_index(pc) -> None:
        import time
        existing = [idx.name for idx in pc.list_indexes()]
        if PINECONE_INDEX_NAME in existing:
            return

        from pinecone import ServerlessSpec
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' (~60s)...", flush=True)
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status.ready:
            time.sleep(2)
        print("Index ready.")

    @staticmethod
    def _chunk(text: str) -> list[tuple[int, int, str]]:
        words = text.split()
        step = max(1, CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS)
        chunks: list[tuple[int, int, str]] = []
        i = 0
        while i < len(words):
            end = min(i + CHUNK_SIZE_WORDS, len(words))
            chunks.append((i, end, " ".join(words[i:end])))
            if end >= len(words):
                break
            i += step
        return chunks
