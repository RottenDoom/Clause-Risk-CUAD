"""
agent/precedent_retrieval.py  —  Embedding Use 2  —  Issue 7

Queries ChromaDB for similar and contrasting precedents.
No access to the raw contract text — receives only extracted_clause_text.

Interface:
    retrieve_precedents(
        extracted_clause_text, clause_family, chroma_client, anthropic_client
    ) -> (list[SimilarPrecedent], list[ContrastingPrecedent])
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    COLLECTION_NAMES,
    CONTRAST_CLAUSE_TRUNCATION_CHARS,
    MAX_TOKENS,
    MODEL,
    PRECEDENT_CONTRAST_FETCH_K,
    PRECEDENT_CONTRAST_RETURN_K,
    PRECEDENT_SIMILAR_TOP_K,
)
from agent.embedder import embed_one
from agent.models import ContrastingPrecedent, SimilarPrecedent


# ---------------------------------------------------------------------------
# Similar precedents
# ---------------------------------------------------------------------------

def _get_similar(
    extracted_text: str,
    family: str,
    chroma_client,
    anthropic_client,
) -> list[SimilarPrecedent]:
    collection = chroma_client.get_collection(COLLECTION_NAMES[family])
    query_emb = embed_one(extracted_text).tolist()

    # Over-fetch to account for deduplication by contract_id
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=min(PRECEDENT_SIMILAR_TOP_K * 3, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # Deduplicate: keep best-scoring chunk per contract_id
    seen: dict[str, dict] = {}
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        cid = meta["contract_id"]
        if cid not in seen or dist < seen[cid]["dist"]:
            seen[cid] = {"meta": meta, "doc": doc, "dist": dist}

    top = sorted(seen.values(), key=lambda x: x["dist"])[:PRECEDENT_SIMILAR_TOP_K]
    if not top:
        return []

    similar: list[SimilarPrecedent] = []
    for item in top:
        meta = item["meta"]
        full_text = meta.get("full_clause_text", item["doc"])
        why = _why_similar(extracted_text, meta["contract_id"], full_text, anthropic_client)
        similar.append(SimilarPrecedent(contract_id=meta["contract_id"], why_similar=why))
    return similar


def _why_similar(
    uploaded_text: str,
    ref_contract_id: str,
    ref_full_text: str,
    anthropic_client,
) -> str:
    prompt = (
        f"UPLOADED CLAUSE:\n{uploaded_text[:CONTRAST_CLAUSE_TRUNCATION_CHARS]}\n\n"
        f"REFERENCE CLAUSE (from contract {ref_contract_id}):\n"
        f"{ref_full_text[:CONTRAST_CLAUSE_TRUNCATION_CHARS]}\n\n"
        "In one sentence, explain why these two clauses are similar in wording, "
        "obligation structure, or business effect."
    )
    try:
        resp = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=128,
            system="You are a contract clause analyst. Be concise.",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception:
        return "Similar in structure and obligation to the uploaded clause."


# ---------------------------------------------------------------------------
# Contrasting precedents
# ---------------------------------------------------------------------------

def _get_contrasting(
    extracted_text: str,
    family: str,
    chroma_client,
    anthropic_client,
) -> list[ContrastingPrecedent]:
    collection = chroma_client.get_collection(COLLECTION_NAMES[family])
    query_emb = embed_one(extracted_text).tolist()

    n_fetch = min(PRECEDENT_CONTRAST_FETCH_K, collection.count())
    if n_fetch == 0:
        return []

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n_fetch,
        include=["documents", "metadatas"],
    )

    # Deduplicate by contract_id, keep one entry per contract
    seen: dict[str, dict] = {}
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        cid = meta["contract_id"]
        if cid not in seen:
            seen[cid] = {"meta": meta, "doc": doc}

    candidates = list(seen.values())
    if not candidates:
        return []

    # Single Claude call to select the most risk-posture-contrasting entries
    candidates_block = "\n".join(
        f"[{i + 1}] contract_id: {c['meta']['contract_id']} | "
        f"text: {c['meta'].get('full_clause_text', c['doc'])[:CONTRAST_CLAUSE_TRUNCATION_CHARS]}"
        for i, c in enumerate(candidates)
    )

    prompt = (
        f"UPLOADED CLAUSE:\n{extracted_text[:CONTRAST_CLAUSE_TRUNCATION_CHARS]}\n\n"
        f"CANDIDATE REFERENCE CLAUSES:\n{candidates_block}\n\n"
        f"From these candidates, select the {PRECEDENT_CONTRAST_RETURN_K} clauses "
        "that have the most materially different risk posture from the uploaded clause. "
        "For each, explain in one sentence why the uploaded clause looks riskier, softer, "
        "or structurally different.\n\n"
        "Respond in JSON (array only, no markdown fences):\n"
        '[{"contract_id": "...", "why_contrasting": "..."}, ...]'
    )
    try:
        resp = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=512,
            system="You are a contract clause risk analyst. Respond only in JSON.",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
        parsed: list[dict] = json.loads(raw)
        return [
            ContrastingPrecedent(
                contract_id=item["contract_id"],
                why_contrasting=item["why_contrasting"],
            )
            for item in parsed
            if item.get("contract_id") and item.get("why_contrasting")
        ]
    except Exception:
        # Graceful degradation: return first candidate with a generic note
        if candidates:
            return [
                ContrastingPrecedent(
                    contract_id=candidates[0]["meta"]["contract_id"],
                    why_contrasting="Risk posture differs from the uploaded clause.",
                )
            ]
        return []


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def retrieve_precedents(
    extracted_clause_text: str,
    clause_family: str,
    chroma_client,
    anthropic_client,
) -> tuple[list[SimilarPrecedent], list[ContrastingPrecedent]]:
    """
    Returns (similar_precedents, contrasting_precedents).
    Both lists may be empty only if the collection has no entries for this family.
    """
    try:
        collection_count = chroma_client.get_collection(
            COLLECTION_NAMES[clause_family]
        ).count()
    except Exception:
        return [], []

    if collection_count == 0:
        return [], []

    similar = _get_similar(extracted_clause_text, clause_family, chroma_client, anthropic_client)
    contrasting = _get_contrasting(extracted_clause_text, clause_family, chroma_client, anthropic_client)
    return similar, contrasting
