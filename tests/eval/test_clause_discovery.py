import pytest
from scripts.eval.loader import EvalRecord


def _make_record(gt_present, clause_found, discovery_score=0.5):
    return EvalRecord(
        contract_id="C1", family="assignment",
        gt_clause_present=gt_present, gt_clause_text="text" if gt_present else None,
        clause_found=clause_found, extracted_clause_text=None,
        structured_interpretation=None, similar_contract_ids=[],
        llm_generated_risk_rating=None, risk_rationale=None,
        discovery_score=discovery_score,
    )


# TP=3, FP=1, FN=1, TN=1  (6 records)
# FN has score=0.0 so no threshold > 0 can recover it — P@80R stays None
RECORDS = [
    _make_record(True,  True,  0.7),   # TP
    _make_record(True,  True,  0.6),   # TP
    _make_record(True,  True,  0.5),   # TP
    _make_record(False, True,  0.4),   # FP
    _make_record(True,  False, 0.0),   # FN — score=0.0, unrecoverable at any positive threshold
    _make_record(False, False, 0.1),   # TN
]


def test_accuracy():
    from scripts.eval.clause_discovery import compute_metrics
    m = compute_metrics(RECORDS)
    assert m.accuracy == pytest.approx(4 / 6)


def test_precision():
    from scripts.eval.clause_discovery import compute_metrics
    m = compute_metrics(RECORDS)
    assert m.precision == pytest.approx(3 / 4)  # TP/(TP+FP)


def test_recall():
    from scripts.eval.clause_discovery import compute_metrics
    m = compute_metrics(RECORDS)
    assert m.recall == pytest.approx(3 / 4)  # TP/(TP+FN)


def test_f1():
    from scripts.eval.clause_discovery import compute_metrics
    m = compute_metrics(RECORDS)
    p, r = 3/4, 3/4
    assert m.f1 == pytest.approx(2 * p * r / (p + r))


def test_f2():
    from scripts.eval.clause_discovery import compute_metrics
    m = compute_metrics(RECORDS)
    p, r = 3/4, 3/4
    assert m.f2 == pytest.approx(5 * p * r / (4 * p + r))


def test_laziness_rate():
    from scripts.eval.clause_discovery import compute_metrics
    m = compute_metrics(RECORDS)
    assert m.laziness_rate == pytest.approx(1 - 3/4)


def test_precision_at_80_recall():
    from scripts.eval.clause_discovery import compute_metrics
    # With 4 actual positives, TP=3 → recall=0.75 < 0.80. P@80R should be None.
    m = compute_metrics(RECORDS)
    assert m.precision_at_80_recall is None


def test_precision_at_80_recall_achievable():
    from scripts.eval.clause_discovery import compute_metrics
    # 5 positives; 4 TPs at high scores → recall can reach 0.80
    records = [
        _make_record(True,  True,  0.9),   # TP
        _make_record(True,  True,  0.8),   # TP
        _make_record(True,  True,  0.75),  # TP
        _make_record(True,  True,  0.6),   # TP
        _make_record(True,  False, 0.2),   # FN (below any threshold we'd pick)
        _make_record(False, False, 0.1),   # TN
    ]
    m = compute_metrics(records)
    # Recall = 4/5 = 0.80 exactly at threshold=0.6; precision at that point = 4/4 = 1.0
    assert m.precision_at_80_recall is not None
    assert m.precision_at_80_recall == pytest.approx(1.0)


def test_empty_records():
    from scripts.eval.clause_discovery import compute_metrics
    m = compute_metrics([])
    assert m.accuracy == 0.0
    assert m.precision == 0.0
    assert m.recall == 0.0
