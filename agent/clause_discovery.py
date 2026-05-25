"""
agent/clause_discovery.py  —  Embedding Use 1

Given a raw contract text and a clause family, finds the most relevant text
span in the contract using an ensemble of anchor queries. No ChromaDB access.

Interface:
    discover_clause(contract_text, clause_family)
        -> (clause_found: bool, extracted_text: str | None, best_score: float)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    CLAUSE_FAMILIES,
    DISCOVERY_MIN_SCORE,
    DISCOVERY_TOP_K,
)
from agent.embedder import embed

# ---------------------------------------------------------------------------
# Anchor queries
# ---------------------------------------------------------------------------
# Five representative sentences per family. Scores are averaged across all
# anchors per chunk, so no single phrasing dominates — more robust to unusual
# legal boilerplate than a single query.

ANCHOR_QUERIES: dict[str, list[str]] = {
    "assignment": [
        "neither party may assign this agreement without prior written consent",
        "assignment of rights and obligations under this contract is prohibited",
        "this agreement shall not be assigned or transferred without approval",
        "a party may assign its rights with the other party's written consent",
        "assignment to affiliates or successors in interest is permitted",
    ],
    "change_of_control": [
        "a change of control event shall trigger termination rights",
        "if either party undergoes a change of control the other party may terminate",
        "change in ownership or control requires prior written notice",
        "in the event of merger acquisition or change of control",
        "the agreement may be terminated upon a change of control of either party",
    ],
    "termination": [
        "either party may terminate this agreement for convenience upon written notice",
        "this agreement may be terminated by either party without cause",
        "termination upon material breach with a cure period",
        "the agreement shall terminate automatically upon insolvency or bankruptcy",
        "notice of termination must be provided at least thirty days in advance",
    ],
    "exclusivity": [
        "the parties agree that this arrangement is exclusive during the term",
        "neither party shall engage with competing vendors or service providers",
        "exclusivity is granted for the duration of this agreement",
        "non-compete restrictions apply within the defined geographic territory",
        "during the term the customer shall not purchase similar services from third parties",
    ],
}


# ---------------------------------------------------------------------------
# Chunking (mirrors build_index.py — same parameters, same logic)
# ---------------------------------------------------------------------------

def _chunk_contract(text: str) -> list[tuple[int, int, str]]:
    """Returns list of (start_word_idx, end_word_idx, chunk_text)."""
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
# Merge selected chunks into a single contiguous span
# ---------------------------------------------------------------------------

def _merge_chunks(
    selected: list[tuple[int, int, str]], all_words: list[str]
) -> str:
    """
    Sort selected chunks by document position, then reconstruct the text
    from the union of their word ranges. Handles overlapping chunks naturally
    since we reconstruct directly from the word list.
    """
    if not selected:
        return ""
    sorted_chunks = sorted(selected, key=lambda c: c[0])
    start = sorted_chunks[0][0]
    end = sorted_chunks[-1][1]
    return " ".join(all_words[start:end])


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def discover_clause(
    contract_text: str,
    clause_family: str,
) -> tuple[bool, str | None, float]:
    """
    Find the clause text for a given family within a raw contract.

    Returns:
        clause_found:         True if the best averaged score >= DISCOVERY_MIN_SCORE
        extracted_clause_text: Merged span of the top-K chunks, or None
        best_score:           Max averaged anchor score across all chunks
    """
    if clause_family not in CLAUSE_FAMILIES:
        raise ValueError(f"Unknown clause family: {clause_family!r}")

    anchors = ANCHOR_QUERIES[clause_family]
    chunks = _chunk_contract(contract_text)
    if not chunks:
        return False, None, 0.0

    chunk_texts = [c[2] for c in chunks]

    # Embed all chunks and all anchor queries in two batches
    chunk_embs = embed(chunk_texts)          # (n_chunks, dim)
    anchor_embs = embed(anchors)             # (n_anchors, dim)

    # Similarity matrix: (n_anchors, n_chunks); dot product == cosine since L2-normalised
    sim_matrix = anchor_embs @ chunk_embs.T  # (n_anchors, n_chunks)

    # Average across anchors to get a single score per chunk
    avg_scores: np.ndarray = sim_matrix.mean(axis=0)  # (n_chunks,)

    best_score = float(avg_scores.max())
    if best_score < DISCOVERY_MIN_SCORE:
        return False, None, best_score

    # Take top-K chunks by averaged score
    k = min(DISCOVERY_TOP_K, len(chunks))
    top_k_indices = np.argsort(avg_scores)[-k:]
    top_k_chunks = [chunks[i] for i in top_k_indices]

    all_words = contract_text.split()
    extracted = _merge_chunks(top_k_chunks, all_words)

    return True, extracted, best_score
