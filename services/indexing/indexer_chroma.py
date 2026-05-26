"""
services/indexing/indexer_chroma.py

ChromaDB indexing service — original implementation, kept for reference.
The active indexer is services/indexing/indexer.py (Pinecone).
"""

import json
from pathlib import Path

from config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS, CLAUSE_FAMILIES, COLLECTION_NAMES


class Indexer:
    """
    Offline indexing service. Reads reference_annotations.json,
    chunks each clause, embeds with sentence-transformers, and
    upserts into four ChromaDB collections (one per clause family).
    """

    def __init__(self, db_path: str, embedding_model: str) -> None:
        self.db_path = db_path
        self.embedding_model = embedding_model

    def build(self, annotations_path: Path, reference_dir: Path) -> dict[str, int]:  # noqa: ARG002
        """
        Build the full index from reference annotations.
        Returns {family: document_count} upserted per collection.
        Re-running is safe — ChromaDB upsert is idempotent.
        """
        import chromadb
        from agent.embedder import embed

        with open(annotations_path, encoding="utf-8") as f:
            annotations: dict[str, dict[str, str | None]] = json.load(f)

        client = chromadb.PersistentClient(path=self.db_path)
        collections = {
            family: client.get_or_create_collection(
                name=COLLECTION_NAMES[family],
                metadata={"hnsw:space": "cosine"},
            )
            for family in CLAUSE_FAMILIES
        }

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

            ids, embeddings_list, docs, metas = [], [], [], []
            seen_counts: dict[str, int] = {}
            for global_idx, record in items:
                cid = record["contract_id"]
                chunk_idx = seen_counts.get(cid, 0)
                seen_counts[cid] = chunk_idx + 1

                ids.append(f"{cid}__{family}__{chunk_idx}")
                embeddings_list.append(all_embeddings[global_idx].tolist())
                docs.append(record["text"])
                metas.append({
                    "contract_id": cid,
                    "clause_family": family,
                    "full_clause_text": record["full_clause_text"],
                    "chunk_index": chunk_idx,
                })

            collections[family].upsert(
                ids=ids, embeddings=embeddings_list, documents=docs, metadatas=metas
            )
            counts[family] = len(ids)

        return counts

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
