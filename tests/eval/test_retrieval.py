import math
import pytest
from scripts.eval.loader import EvalRecord


def _rec(similar_ids, relevant_ids, extracted_text="clause text", score=0.5):
    return EvalRecord(
        contract_id="C1", family="assignment",
        gt_clause_present=True, gt_clause_text="text",
        clause_found=True, extracted_clause_text=extracted_text,
        structured_interpretation=None,
        similar_contract_ids=list(similar_ids),
        llm_generated_risk_rating=None, risk_rationale=None,
        discovery_score=score,
        relevant_reference_ids=frozenset(relevant_ids),
    )


def test_recall_at_3_hit():
    from scripts.eval.retrieval import compute_metrics
    # Retrieved [A, B, C], relevant = {B} → hit in top-3
    rec = _rec(["A", "B", "C"], ["B"])
    m = compute_metrics([rec], ref_annotations={})
    assert m.recall_at_3 == pytest.approx(1.0)


def test_recall_at_3_miss():
    from scripts.eval.retrieval import compute_metrics
    rec = _rec(["A", "B", "C"], ["D"])
    m = compute_metrics([rec], ref_annotations={})
    assert m.recall_at_3 == pytest.approx(0.0)


def test_mrr_first_hit():
    from scripts.eval.retrieval import compute_metrics
    rec = _rec(["A", "B", "C"], ["A"])  # first result is relevant → MRR = 1/1
    m = compute_metrics([rec], ref_annotations={})
    assert m.mrr == pytest.approx(1.0)


def test_mrr_second_hit():
    from scripts.eval.retrieval import compute_metrics
    rec = _rec(["A", "B", "C"], ["B"])  # second result relevant → MRR = 1/2
    m = compute_metrics([rec], ref_annotations={})
    assert m.mrr == pytest.approx(0.5)


def test_mrr_no_hit():
    from scripts.eval.retrieval import compute_metrics
    rec = _rec(["A", "B", "C"], ["D"])
    m = compute_metrics([rec], ref_annotations={})
    assert m.mrr == pytest.approx(0.0)


def test_ndcg_at_3_perfect():
    from scripts.eval.retrieval import compute_metrics
    rec = _rec(["A", "B", "C"], ["A", "B", "C"])  # all 3 relevant
    m = compute_metrics([rec], ref_annotations={})
    assert m.ndcg_at_3 == pytest.approx(1.0)


def test_ndcg_at_3_partial():
    from scripts.eval.retrieval import compute_metrics
    # Relevant = {A, C}; retrieved = [A, B, C]
    # DCG = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.631 = 1.631
    rec = _rec(["A", "B", "C"], ["A", "C"])
    m = compute_metrics([rec], ref_annotations={})
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    assert m.ndcg_at_3 == pytest.approx(dcg / idcg, rel=1e-3)


def test_skips_clause_not_found():
    from scripts.eval.retrieval import compute_metrics
    # clause_found=False → should be skipped
    rec = EvalRecord(
        contract_id="C1", family="assignment",
        gt_clause_present=False, gt_clause_text=None,
        clause_found=False, extracted_clause_text=None,
        structured_interpretation=None, similar_contract_ids=[],
        llm_generated_risk_rating=None, risk_rationale=None,
        relevant_reference_ids=frozenset(),
    )
    m = compute_metrics([rec], ref_annotations={})
    assert m.n_queries == 0


def test_empty():
    from scripts.eval.retrieval import compute_metrics
    m = compute_metrics([], ref_annotations={})
    assert m.n_queries == 0
    assert m.recall_at_3 == 0.0
