"""
scripts/build_index.py

Offline indexing step. Reads reference_annotations.json, chunks and embeds
each annotated clause text, and upserts into four ChromaDB collections —
one per clause family.

Run once after prepare_data.py. Re-running is safe (upsert is idempotent).

Usage:
    python3 scripts/build_index.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CHROMA_DB_PATH,
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    CLAUSE_FAMILIES,
    COLLECTION_NAMES,
    REFERENCE_ANNOTATIONS_PATH,
)
from agent.embedder import embed


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str) -> list[tuple[int, int, str]]:
    """
    Word-level sliding window chunker.
    Returns list of (start_word_idx, end_word_idx, chunk_text).
    Using word-level rather than token-level avoids a tokenizer dependency here;
    at ~1.3 words/token for legal text the approximation is close enough.
    """
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


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_index() -> None:
    if not REFERENCE_ANNOTATIONS_PATH.exists():
        sys.exit(
            f"ERROR: {REFERENCE_ANNOTATIONS_PATH} not found.\n"
            "  Run scripts/prepare_data.py first."
        )

    with open(REFERENCE_ANNOTATIONS_PATH, encoding="utf-8") as f:
        annotations: dict[str, dict[str, str | None]] = json.load(f)

    print(f"Loaded {len(annotations)} reference contracts.")

    try:
        import chromadb
    except ImportError:
        sys.exit("ERROR: chromadb not installed. Run: uv pip install chromadb")

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    collections = {
        family: client.get_or_create_collection(
            name=COLLECTION_NAMES[family],
            metadata={"hnsw:space": "cosine"},
        )
        for family in CLAUSE_FAMILIES
    }

    # --- Pass 1: collect every chunk before touching the embedder ---
    # counter[contract_id][family] avoids the O(n²) list scan that would
    # otherwise stall here for several seconds before any output appears.
    pending: list[dict] = []
    counter: dict[str, dict[str, int]] = {}

    print("Collecting chunks", end="", flush=True)
    for contract_id, family_texts in annotations.items():
        counter[contract_id] = {}
        for family in CLAUSE_FAMILIES:
            clause_text = family_texts.get(family)
            if not clause_text or not clause_text.strip():
                continue
            counter[contract_id][family] = 0
            for _start, _end, chunk_text_str in chunk_text(clause_text):
                idx = counter[contract_id][family]
                counter[contract_id][family] += 1
                pending.append({
                    "family": family,
                    "contract_id": contract_id,
                    "chunk_idx": idx,
                    "text": chunk_text_str,
                    "full_clause_text": clause_text,
                })
        print(".", end="", flush=True)
    print(f" {len(pending)} chunks total")

    # Model download (~90 MB) happens here on first run — no progress bar from
    # the library, so we print explicitly so it doesn't look frozen.
    print("Loading embedding model (downloads ~90 MB on first run)...", flush=True)
    all_texts = [p["text"] for p in pending]
    all_embeddings = embed(all_texts, show_progress=True)  # tqdm bar during inference

    # --- Pass 2: upsert into ChromaDB per family ---
    print("\nUpserting into ChromaDB...")
    total_chunks = {family: 0 for family in CLAUSE_FAMILIES}

    # Group pending records by family for efficient upsert
    by_family: dict[str, list[tuple[int, dict]]] = {f: [] for f in CLAUSE_FAMILIES}
    for i, record in enumerate(pending):
        by_family[record["family"]].append((i, record))

    for family in CLAUSE_FAMILIES:
        items = by_family[family]
        if not items:
            continue

        # Assign stable chunk indices per (contract, family) pair
        seen: dict[str, int] = {}
        ids, embeddings_list, docs, metas = [], [], [], []
        for global_idx, record in items:
            cid = record["contract_id"]
            chunk_idx = seen.get(cid, 0)
            seen[cid] = chunk_idx + 1

            ids.append(f"{cid}__{family}__{chunk_idx}")
            embeddings_list.append(all_embeddings[global_idx].tolist())
            docs.append(record["text"])
            metas.append({
                "contract_id": cid,
                "clause_family": family,
                "full_clause_text": record["full_clause_text"],
                "chunk_index": chunk_idx,
            })

        collections[family].upsert(ids=ids, embeddings=embeddings_list, documents=docs, metadatas=metas)
        total_chunks[family] = len(ids)
        print(f"  {COLLECTION_NAMES[family]:<32} {len(ids):>4} chunks")

    print(f"\nIndex persisted at: {CHROMA_DB_PATH}")
    print(f"Total chunks indexed: {sum(total_chunks.values())}")
    print("Done. Next: python3 scripts/run_review.py --contract <path>")


if __name__ == "__main__":
    build_index()
