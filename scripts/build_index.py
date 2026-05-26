"""
scripts/build_index.py

Offline indexing step. Reads reference_annotations.json, chunks and embeds
each annotated clause text, and upserts into the selected vector store(s).

Backends
--------
  --chromadb   Index into local ChromaDB  (data/chroma_db/)
  --pinecone   Index into Pinecone cloud  (index: cuad-contracts)
  (no flag)    Run both sequentially

The embedding step is shared — text is embedded once regardless of how many
backends are selected.

Parallelise across backends by running two processes simultaneously:
    python3 scripts/build_index.py --chromadb &
    python3 scripts/build_index.py --pinecone &

Usage:
    python3 scripts/build_index.py               # both backends
    python3 scripts/build_index.py --pinecone    # Pinecone only
    python3 scripts/build_index.py --chromadb    # ChromaDB only
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CHROMA_DB_PATH,
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    CLAUSE_FAMILIES,
    COLLECTION_NAMES,
    EMBEDDING_DIM,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
    REFERENCE_ANNOTATIONS_PATH,
)
from agent.embedder import embed


# ---------------------------------------------------------------------------
# Shared: chunking + data loading
# ---------------------------------------------------------------------------

def chunk_text(text: str) -> list[tuple[int, int, str]]:
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


def load_and_embed() -> tuple[list[dict], object]:
    """
    Load annotations, collect all chunks, embed in one batch.
    Returns (pending_records, all_embeddings).
    """
    if not REFERENCE_ANNOTATIONS_PATH.exists():
        sys.exit(
            f"ERROR: {REFERENCE_ANNOTATIONS_PATH} not found.\n"
            "  Run scripts/prepare_data.py first."
        )

    with open(REFERENCE_ANNOTATIONS_PATH, encoding="utf-8") as f:
        annotations: dict[str, dict[str, str | None]] = json.load(f)

    print(f"Loaded {len(annotations)} reference contracts.")

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
            for _start, _end, chunk_str in chunk_text(clause_text):
                idx = counter[contract_id][family]
                counter[contract_id][family] += 1
                pending.append({
                    "family": family,
                    "contract_id": contract_id,
                    "chunk_idx": idx,
                    "text": chunk_str,
                    "full_clause_text": clause_text,
                })
        print(".", end="", flush=True)
    print(f" {len(pending)} chunks total")

    print("Loading embedding model (downloads ~90 MB on first run)...", flush=True)
    all_embeddings = embed([p["text"] for p in pending], show_progress=True)

    return pending, all_embeddings


def group_by_family(pending: list[dict]) -> dict[str, list[tuple[int, dict]]]:
    by_family: dict[str, list[tuple[int, dict]]] = {f: [] for f in CLAUSE_FAMILIES}
    for i, record in enumerate(pending):
        by_family[record["family"]].append((i, record))
    return by_family


# ---------------------------------------------------------------------------
# Backend: ChromaDB
# ---------------------------------------------------------------------------

def build_chromadb(pending: list[dict], all_embeddings) -> None:
    print("\n─── ChromaDB ───")
    try:
        import chromadb
    except ImportError:
        print("ERROR: chromadb not installed. Run: uv pip install chromadb")
        return

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collections = {
        family: client.get_or_create_collection(
            name=COLLECTION_NAMES[family],
            metadata={"hnsw:space": "cosine"},
        )
        for family in CLAUSE_FAMILIES
    }

    by_family = group_by_family(pending)
    total: dict[str, int] = {}

    for family in CLAUSE_FAMILIES:
        items = by_family[family]
        if not items:
            total[family] = 0
            continue

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
        total[family] = len(ids)
        print(f"  {COLLECTION_NAMES[family]:<32} {len(ids):>4} chunks")

    print(f"ChromaDB persisted at: {CHROMA_DB_PATH}")
    print(f"Total chunks: {sum(total.values())}")


# ---------------------------------------------------------------------------
# Backend: Pinecone
# ---------------------------------------------------------------------------

_PINECONE_BATCH = 100
_METADATA_TEXT_LIMIT = 20_000  # 20 KB — well under Pinecone's 40 KB limit


def _ensure_pinecone_index(pc) -> None:
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME in existing:
        return

    from pinecone import ServerlessSpec
    print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' (may take ~60s)...", flush=True)
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status.ready:
        time.sleep(2)
    print("Index ready.")


def build_pinecone(pending: list[dict], all_embeddings) -> None:
    import os
    print("\n─── Pinecone ───")

    try:
        from pinecone import Pinecone
    except ImportError:
        print("ERROR: pinecone not installed. Run: uv pip install 'pinecone>=3.0.0'")
        return

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY not set. Add it to .env and retry.")
        return

    pc = Pinecone(api_key=api_key)
    _ensure_pinecone_index(pc)
    index = pc.Index(PINECONE_INDEX_NAME)

    by_family = group_by_family(pending)
    total: dict[str, int] = {}

    for family in CLAUSE_FAMILIES:
        items = by_family[family]
        if not items:
            total[family] = 0
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

        for i in range(0, len(vectors), _PINECONE_BATCH):
            index.upsert(vectors=vectors[i:i + _PINECONE_BATCH], namespace=family)

        total[family] = len(vectors)
        print(f"  {COLLECTION_NAMES[family]:<32} {len(vectors):>4} vectors → namespace '{family}'")

    print(f"Pinecone index '{PINECONE_INDEX_NAME}' populated.")
    print(f"Total vectors: {sum(total.values())}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build vector index from reference contract annotations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 scripts/build_index.py                 # both backends\n"
            "  python3 scripts/build_index.py --pinecone      # Pinecone only\n"
            "  python3 scripts/build_index.py --chromadb      # ChromaDB only\n"
            "\n"
            "Parallel (fastest):\n"
            "  python3 scripts/build_index.py --chromadb &\n"
            "  python3 scripts/build_index.py --pinecone &\n"
            "  wait"
        ),
    )
    parser.add_argument("--pinecone", action="store_true", help="Index into Pinecone")
    parser.add_argument("--chromadb", action="store_true", help="Index into ChromaDB")
    args = parser.parse_args()

    # If neither flag is given, run both
    run_pinecone = args.pinecone or (not args.pinecone and not args.chromadb)
    run_chromadb = args.chromadb or (not args.pinecone and not args.chromadb)

    pending, all_embeddings = load_and_embed()

    if run_chromadb:
        build_chromadb(pending, all_embeddings)
    if run_pinecone:
        build_pinecone(pending, all_embeddings)

    print("\nDone. Next: python3 scripts/run_review.py --contract <path>")


if __name__ == "__main__":
    main()
