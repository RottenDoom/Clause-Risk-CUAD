"""
agent/clause_discovery.py  —  Embedding Use 1

Given a raw contract text and a clause family, finds the most relevant text
span in the contract using an ensemble of anchor queries. No ChromaDB access.

Interface:
    discover_clause(contract_text, clause_family)
        -> (clause_found: bool, extracted_text: str | None, best_score: float)
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

logger = logging.getLogger(__name__)
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

# Broader anchor sets used on retry when the standard set scores below threshold.
# More varied phrasing catches contracts that use unusual boilerplate.
BROAD_ANCHOR_QUERIES: dict[str, list[str]] = {
    "assignment": [
        "transfer of rights obligations or interests under this agreement",
        "consent is required before any assignment or delegation",
        "the agreement binds the parties their successors and permitted assigns",
        "assignment without consent shall be void and of no effect",
        "party may not delegate its duties without written permission",
        "change of ownership does not automatically assign this contract",
        "permitted assignments include transfers to wholly owned subsidiaries",
    ],
    "change_of_control": [
        "direct or indirect change in majority ownership of a party",
        "sale of all or substantially all assets triggers special rights",
        "upon a corporate restructuring or reorganization the other party must be notified",
        "change in beneficial ownership of voting shares",
        "surviving entity in a merger shall assume all obligations",
        "party undergoing a change of control must provide advance written notice",
        "change of control event gives the non-affected party termination rights",
    ],
    "termination": [
        "this agreement may be terminated by written notice to the other party",
        "either party may end this agreement upon sixty days prior written notice",
        "termination shall not relieve either party of accrued obligations",
        "upon expiration or earlier termination of this agreement",
        "the agreement shall automatically expire unless renewed",
        "party may terminate immediately upon material uncured breach",
        "termination rights arise upon insolvency liquidation or dissolution",
    ],
    "exclusivity": [
        "customer shall obtain the service exclusively from provider",
        "supplier is the sole and exclusive source for the products",
        "during the term neither party may work with competitors",
        "non-solicitation and non-competition obligations survive termination",
        "exclusive license granted for the territory and term",
        "restrictions on engaging with third-party vendors for similar services",
        "preferred supplier status grants the right of first refusal",
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
    broad: bool = False,
) -> tuple[bool, str | None, float]:
    """
    Find the clause text for a given family within a raw contract.

    Args:
        broad: If True, use the extended BROAD_ANCHOR_QUERIES set and relax the
               score threshold by 15%. Called on retry by loop.py when the first
               pass scores below DISCOVERY_MIN_SCORE.

    Returns:
        clause_found:          True if the best averaged score >= threshold
        extracted_clause_text: Merged span of the top-K chunks, or None
        best_score:            Max averaged anchor score across all chunks
    """
    if clause_family not in CLAUSE_FAMILIES:
        raise ValueError(f"Unknown clause family: {clause_family!r}")

    tid = threading.get_ident()
    t_entry = time.monotonic()
    anchors = BROAD_ANCHOR_QUERIES[clause_family] if broad else ANCHOR_QUERIES[clause_family]
    chunks = _chunk_contract(contract_text)
    logger.info(
        "discover[t%d] family=%s broad=%s chunked n_chunks=%d (%.2fs)",
        tid, clause_family, broad, len(chunks), time.monotonic() - t_entry,
    )
    if not chunks:
        return False, None, 0.0

    chunk_texts = [c[2] for c in chunks]

    # Embed all chunks and all anchor queries in two batches
    t_emb = time.monotonic()
    chunk_embs = embed(chunk_texts)          # (n_chunks, dim)
    logger.info("discover[t%d] family=%s embedded chunks (%.2fs)",
                tid, clause_family, time.monotonic() - t_emb)
    t_emb2 = time.monotonic()
    anchor_embs = embed(anchors)             # (n_anchors, dim)
    logger.info("discover[t%d] family=%s embedded anchors (%.2fs)",
                tid, clause_family, time.monotonic() - t_emb2)

    # Similarity matrix: (n_anchors, n_chunks); dot product == cosine since L2-normalised
    sim_matrix = anchor_embs @ chunk_embs.T  # (n_anchors, n_chunks)

    # Average across anchors to get a single score per chunk
    avg_scores: np.ndarray = sim_matrix.mean(axis=0)  # (n_chunks,)

    best_score = float(avg_scores.max())
    threshold = DISCOVERY_MIN_SCORE * 0.85 if broad else DISCOVERY_MIN_SCORE

    if best_score < threshold:
        logger.info(
            "family=%s broad=%s score=%.3f threshold=%.3f found=False",
            clause_family, broad, best_score, threshold,
        )
        return False, None, best_score

    # Take top-K chunks by averaged score
    k = min(DISCOVERY_TOP_K, len(chunks))
    top_k_indices = np.argsort(avg_scores)[-k:]
    top_k_chunks = [chunks[i] for i in top_k_indices]

    all_words = contract_text.split()
    extracted = _merge_chunks(top_k_chunks, all_words)

    logger.info(
        "family=%s broad=%s score=%.3f threshold=%.3f found=True chunks=%d top_k=%d",
        clause_family, broad, best_score, threshold, len(chunks), k,
    )
    return True, extracted, best_score
