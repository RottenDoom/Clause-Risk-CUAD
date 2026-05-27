"""scripts/eval/clause_discovery.py — Stage 1 metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from scripts.eval.loader import EvalRecord


@dataclass
class DiscoveryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    f2: float
    laziness_rate: float
    precision_at_80_recall: Optional[float]
    tp: int
    fp: int
    fn: int
    tn: int
    n_records: int


def compute_metrics(records: list[EvalRecord]) -> DiscoveryMetrics:
    if not records:
        return DiscoveryMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, 0, 0, 0, 0, 0)

    tp = sum(1 for r in records if r.gt_clause_present and r.clause_found)
    fp = sum(1 for r in records if not r.gt_clause_present and r.clause_found)
    fn = sum(1 for r in records if r.gt_clause_present and not r.clause_found)
    tn = sum(1 for r in records if not r.gt_clause_present and not r.clause_found)
    n = len(records)

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    f2 = (5 * precision * recall / (4 * precision + recall)
          if (4 * precision + recall) > 0 else 0.0)
    laziness_rate = 1.0 - recall

    p_at_80r = _precision_at_recall(records, target_recall=0.80)

    return DiscoveryMetrics(
        accuracy=accuracy, precision=precision, recall=recall,
        f1=f1, f2=f2, laziness_rate=laziness_rate,
        precision_at_80_recall=p_at_80r,
        tp=tp, fp=fp, fn=fn, tn=tn, n_records=n,
    )


def _precision_at_recall(records: list[EvalRecord], target_recall: float) -> Optional[float]:
    """
    Vary the discovery_score threshold from high to low. At each threshold:
      - predicted_found = discovery_score >= threshold
    Find the lowest threshold where recall >= target_recall and return the precision there.
    Returns None if recall never reaches target_recall.
    """
    n_pos = sum(1 for r in records if r.gt_clause_present)
    if n_pos == 0:
        return None

    thresholds = sorted({r.discovery_score for r in records if r.discovery_score > 0.0}, reverse=True)
    for t in thresholds:
        tp_t = sum(1 for r in records if r.gt_clause_present and r.discovery_score >= t)
        fp_t = sum(1 for r in records if not r.gt_clause_present and r.discovery_score >= t)
        recall_t = tp_t / n_pos
        if recall_t >= target_recall:
            return tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
    return None
