"""scripts/eval/retrieval.py — Stage 2 metrics.

Relevance oracle: binary — a reference contract is relevant if it has a
non-null annotation for the same clause family (derived from reference_annotations.json
via loader.EvalRecord.relevant_reference_ids).

Semantic similarity uses the already-loaded sentence-transformer embedder.
ref_annotations: dict mapping contract_id → {family: clause_text | None}
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from scripts.eval.loader import EvalRecord


@dataclass
class RetrievalMetrics:
    recall_at_3: float
    mrr: float
    ndcg_at_3: float
    avg_semantic_sim: float
    n_queries: int


def compute_metrics(
    records: list[EvalRecord],
    ref_annotations: dict,  # contract_id → {family → text | None}
) -> RetrievalMetrics:
    """Evaluate only records where clause_found=True (retrieval only runs then)."""
    queries = [r for r in records if r.clause_found]
    if not queries:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0.0, 0)

    r3_scores, mrr_scores, ndcg_scores, sim_scores = [], [], [], []

    for rec in queries:
        retrieved = rec.similar_contract_ids
        relevant = rec.relevant_reference_ids

        r3_scores.append(_recall_at_k(retrieved, relevant, k=3))
        mrr_scores.append(_mrr(retrieved, relevant))
        ndcg_scores.append(_ndcg_at_k(retrieved, relevant, k=3))

        sim = _semantic_sim(rec, ref_annotations)
        if sim is not None:
            sim_scores.append(sim)

    return RetrievalMetrics(
        recall_at_3=float(np.mean(r3_scores)),
        mrr=float(np.mean(mrr_scores)),
        ndcg_at_3=float(np.mean(ndcg_scores)),
        avg_semantic_sim=float(np.mean(sim_scores)) if sim_scores else 0.0,
        n_queries=len(queries),
    )


def _recall_at_k(retrieved: list[str], relevant: frozenset, k: int = 3) -> float:
    """Binary hit rate: 1.0 if ≥1 relevant in top-k, else 0.0."""
    return 1.0 if any(doc in relevant for doc in retrieved[:k]) else 0.0


def _mrr(retrieved: list[str], relevant: frozenset) -> float:
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(retrieved: list[str], relevant: frozenset, k: int = 3) -> float:
    """nDCG with binary gains."""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc in enumerate(retrieved[:k], start=1)
        if doc in relevant
    )
    ideal_n = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_n + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _semantic_sim(rec: EvalRecord, ref_annotations: dict) -> float | None:
    """Avg cosine similarity between test clause and retrieved clause texts."""
    if not rec.extracted_clause_text or not rec.similar_contract_ids:
        return None

    ref_texts = [
        ref_annotations.get(cid, {}).get(rec.family)
        for cid in rec.similar_contract_ids
    ]
    ref_texts = [t for t in ref_texts if t]
    if not ref_texts:
        return None

    try:
        from agent.embedder import embed
        all_texts = [rec.extracted_clause_text] + ref_texts
        embs = embed(all_texts)
        query_emb = embs[0]
        ref_embs = embs[1:]
        sims = ref_embs @ query_emb  # dot product == cosine (L2-normalised)
        return float(np.mean(sims))
    except Exception:
        return None
