import pytest
from scripts.eval.loader import EvalRecord


def _rec(family, interp, clause_found=True):
    return EvalRecord(
        contract_id="C1", family=family,
        gt_clause_present=clause_found, gt_clause_text="t" if clause_found else None,
        clause_found=clause_found, extracted_clause_text="t" if clause_found else None,
        structured_interpretation=interp,
        similar_contract_ids=[], llm_generated_risk_rating=None,
        risk_rationale=None, relevant_reference_ids=frozenset(),
    )


FULL_ASSIGNMENT = {
    "assignment_restricted": True, "consent_required": True,
    "restricted_party": "assignor", "exceptions_detected": [],
    "confidence": 0.9,
}

PARTIAL_ASSIGNMENT = {
    "assignment_restricted": True, "consent_required": None,  # None = incomplete
    "restricted_party": "assignor",
    # "exceptions_detected" missing entirely
    "confidence": 0.7,
}


def test_full_completeness():
    from scripts.eval.interpretation import compute_metrics
    rec = _rec("assignment", FULL_ASSIGNMENT)
    m = compute_metrics([rec])
    assert m.field_completeness_rate == pytest.approx(1.0)


def test_partial_completeness():
    from scripts.eval.interpretation import compute_metrics
    rec = _rec("assignment", PARTIAL_ASSIGNMENT)
    m = compute_metrics([rec])
    # 5 expected fields; exceptions_detected key is absent → 4/5 (consent_required=None counts as present)
    assert m.field_completeness_rate == pytest.approx(4 / 5)


def test_skips_clause_not_found():
    from scripts.eval.interpretation import compute_metrics
    rec = _rec("assignment", None, clause_found=False)
    m = compute_metrics([rec])
    assert m.n_evaluated == 0
    assert m.field_completeness_rate == 0.0


def test_per_family_completeness():
    from scripts.eval.interpretation import compute_metrics
    recs = [
        _rec("assignment", FULL_ASSIGNMENT),
        _rec("termination", {"termination_for_convenience": True, "notice_period_days": 30,
                              "cure_period_days": None, "mutual_termination_right": False,
                              "grounds_detected": [], "confidence": 0.8}),
    ]
    m = compute_metrics(recs)
    assert m.per_family_completeness["assignment"] == pytest.approx(1.0)
    assert m.per_family_completeness["termination"] == pytest.approx(1.0)
