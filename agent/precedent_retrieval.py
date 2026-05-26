"""
agent/precedent_retrieval.py  —  Embedding Use 2

Queries ChromaDB for similar and contrasting precedents via the Retriever service.
No access to the raw contract text — receives only extracted_clause_text.
No direct chromadb or anthropic imports — uses service abstractions only.

Interface:
    retrieve_precedents(
        extracted_clause_text, clause_family, retriever, llm
    ) -> (list[SimilarPrecedent], list[ContrastingPrecedent])

LLM call budget per family (when clause is found):
  - _get_similar:     1 batch call  (all why_similar explanations in one JSON response)
  - _get_contrasting: 1 batch call  (candidate selection + why_contrasting in one call)
  Total: 2 LLM calls per family, 8 calls across all 4 families.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

from config import (
    CONTRAST_CLAUSE_TRUNCATION_CHARS,
    MAX_TOKENS,
    PRECEDENT_CONTRAST_RETURN_K,
    PRECEDENT_SIMILAR_TOP_K,
)
from agent.embedder import embed_one
from agent.models import ContrastingPrecedent, SimilarPrecedent
from services.generation.base import LLMClient
from services.retrieval.retriever import Retriever


# ---------------------------------------------------------------------------
# Similar precedents — single batch call
# ---------------------------------------------------------------------------

def _get_similar(
    extracted_text: str,
    family: str,
    retriever: Retriever,
    llm: LLMClient,
) -> list[SimilarPrecedent]:
    query_emb = embed_one(extracted_text).tolist()
    raw = retriever.query(query_emb, family, n_results=PRECEDENT_SIMILAR_TOP_K)

    # Deduplicate: keep best-scoring (lowest distance) chunk per contract_id
    seen: dict[str, dict] = {}
    for item in raw:
        cid = item["contract_id"]
        dist = item["distance"] or 0.0
        if cid not in seen or dist < seen[cid]["distance"]:
            seen[cid] = item

    top = sorted(seen.values(), key=lambda x: x["distance"] or 0.0)[:PRECEDENT_SIMILAR_TOP_K]
    if not top:
        return []

    # Single batch call: all candidates → one JSON response with all why_similar fields.
    # Avoids N separate round-trips for what is fundamentally one comparison task.
    candidates_block = "\n".join(
        f"[{i + 1}] contract_id: {c['contract_id']} | "
        f"text: {c['full_clause_text'][:CONTRAST_CLAUSE_TRUNCATION_CHARS]}"
        for i, c in enumerate(top)
    )
    prompt = (
        f"UPLOADED CLAUSE:\n{extracted_text[:CONTRAST_CLAUSE_TRUNCATION_CHARS]}\n\n"
        f"REFERENCE CLAUSES:\n{candidates_block}\n\n"
        "For each reference clause, write exactly one sentence explaining why it is "
        "similar to the uploaded clause in wording, obligation structure, or business effect.\n\n"
        "Respond in JSON (array only, no markdown fences):\n"
        '[{"contract_id": "...", "why_similar": "..."}, ...]'
    )
    try:
        parsed: list[dict] = llm.generate_json(
            prompt,
            system="You are a contract clause analyst. Respond only in JSON.",
            max_tokens=512,
        )
        result = [
            SimilarPrecedent(
                contract_id=item["contract_id"],
                why_similar=item["why_similar"],
            )
            for item in parsed
            if item.get("contract_id") and item.get("why_similar")
        ]
        if result:
            return result
    except Exception as exc:
        logger.warning("family=%s why_similar batch call failed — using fallback: %s", family, exc)

    # Fallback: return candidates with a generic reason rather than failing silently
    return [
        SimilarPrecedent(
            contract_id=c["contract_id"],
            why_similar="Similar in structure and obligation to the uploaded clause.",
        )
        for c in top
    ]


# ---------------------------------------------------------------------------
# Contrasting precedents — single batch call (unchanged, already batched)
# ---------------------------------------------------------------------------

def _get_contrasting(
    extracted_text: str,
    family: str,
    retriever: Retriever,
    llm: LLMClient,
) -> list[ContrastingPrecedent]:
    query_emb = embed_one(extracted_text).tolist()
    raw = retriever.query_for_contrast(query_emb, family)

    # Deduplicate by contract_id
    seen: dict[str, dict] = {}
    for item in raw:
        cid = item["contract_id"]
        if cid not in seen:
            seen[cid] = item

    candidates = list(seen.values())
    if not candidates:
        return []

    candidates_block = "\n".join(
        f"[{i + 1}] contract_id: {c['contract_id']} | "
        f"text: {c['full_clause_text'][:CONTRAST_CLAUSE_TRUNCATION_CHARS]}"
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
        parsed: list[dict] = llm.generate_json(
            prompt,
            system="You are a contract clause risk analyst. Respond only in JSON.",
            max_tokens=MAX_TOKENS,
        )
        result = [
            ContrastingPrecedent(
                contract_id=item["contract_id"],
                why_contrasting=item["why_contrasting"],
            )
            for item in parsed
            if item.get("contract_id") and item.get("why_contrasting")
        ]
        if result:
            return result
    except Exception:
        pass

    return [
        ContrastingPrecedent(
            contract_id=candidates[0]["contract_id"],
            why_contrasting="Risk posture differs from the uploaded clause.",
        )
    ]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def retrieve_precedents(
    extracted_clause_text: str,
    clause_family: str,
    retriever: Retriever,
    llm: LLMClient,
) -> tuple[list[SimilarPrecedent], list[ContrastingPrecedent]]:
    """
    Returns (similar_precedents, contrasting_precedents).
    Both lists may be empty only if the collection has no entries for this family.
    """
    count = retriever.collection_count(clause_family)
    if count == 0:
        logger.warning("family=%s collection empty — skipping retrieval", clause_family)
        return [], []

    similar = _get_similar(extracted_clause_text, clause_family, retriever, llm)
    contrasting = _get_contrasting(extracted_clause_text, clause_family, retriever, llm)
    logger.info("family=%s similar=%d contrasting=%d", clause_family, len(similar), len(contrasting))
    return similar, contrasting
