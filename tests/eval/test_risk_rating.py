import pytest
from scripts.eval.loader import EvalRecord


def _rec(family, interp, pipeline_rating, clause_found=True):
    return EvalRecord(
        contract_id="C1", family=family,
        gt_clause_present=clause_found, gt_clause_text="t" if clause_found else None,
        clause_found=clause_found, extracted_clause_text="t" if clause_found else None,
        structured_interpretation=interp,
        similar_contract_ids=[], llm_generated_risk_rating=pipeline_rating,
        risk_rationale="Some rationale.", relevant_reference_ids=frozenset(),
    )


def test_assignment_high():
    from scripts.eval.risk_rating import heuristic_risk_label
    interp = {"assignment_restricted": True, "consent_required": True,
               "exceptions_detected": [], "restricted_party": "assignor", "confidence": 0.9}
    assert heuristic_risk_label("assignment", interp) == "high"


def test_assignment_medium():
    from scripts.eval.risk_rating import heuristic_risk_label
    interp = {"assignment_restricted": True, "consent_required": True,
               "exceptions_detected": ["affiliates"], "restricted_party": "assignor", "confidence": 0.8}
    assert heuristic_risk_label("assignment", interp) == "medium"


def test_assignment_low():
    from scripts.eval.risk_rating import heuristic_risk_label
    interp = {"assignment_restricted": False, "consent_required": False,
               "exceptions_detected": [], "restricted_party": "neither", "confidence": 0.7}
    assert heuristic_risk_label("assignment", interp) == "low"


def test_termination_high():
    from scripts.eval.risk_rating import heuristic_risk_label
    interp = {"termination_for_convenience": True, "notice_period_days": 15,
               "cure_period_days": None, "mutual_termination_right": False,
               "grounds_detected": [], "confidence": 0.85}
    assert heuristic_risk_label("termination", interp) == "high"


def test_termination_medium():
    from scripts.eval.risk_rating import heuristic_risk_label
    interp = {"termination_for_convenience": True, "notice_period_days": 60,
               "cure_period_days": 30, "mutual_termination_right": True,
               "grounds_detected": [], "confidence": 0.85}
    assert heuristic_risk_label("termination", interp) == "medium"


def test_heuristic_agreement_rate():
    from scripts.eval.risk_rating import compute_heuristic_metrics
    recs = [
        _rec("assignment",
             {"assignment_restricted": True, "consent_required": True,
              "exceptions_detected": [], "restricted_party": "assignor", "confidence": 0.9},
             "high"),  # pipeline=high, heuristic=high → agree
        _rec("assignment",
             {"assignment_restricted": True, "consent_required": True,
              "exceptions_detected": ["affiliates"], "restricted_party": "assignor", "confidence": 0.8},
             "high"),  # pipeline=high, heuristic=medium → disagree
    ]
    m = compute_heuristic_metrics(recs)
    assert m.heuristic_agreement_pct == pytest.approx(0.5)
    assert m.n_compared == 2


def test_skips_not_found():
    from scripts.eval.risk_rating import compute_heuristic_metrics
    rec = _rec("assignment", None, None, clause_found=False)
    m = compute_heuristic_metrics([rec])
    assert m.n_compared == 0


# --- LLM-as-judge tests ---

class MockLLM:
    def generate_json(self, prompt, system="", max_tokens=512):
        return {
            "score": 4,
            "rating_plausible": True,
            "reference_rationale": "Consent required with no exceptions is high risk.",
        }


def test_judge_score_parsed():
    from scripts.eval.risk_rating import run_llm_judge
    rec = _rec("assignment",
               {"assignment_restricted": True, "consent_required": True,
                "exceptions_detected": [], "restricted_party": "assignor", "confidence": 0.9},
               "high")
    rec.risk_rationale = "Strict consent required."
    result = run_llm_judge([rec], llm=MockLLM())
    assert result.avg_judge_score == pytest.approx(4.0)
    assert result.judge_pass_pct == pytest.approx(1.0)
    assert result.n_judged == 1


def test_jaccard_similarity():
    from scripts.eval.risk_rating import _jaccard
    assert _jaccard("the quick brown fox", "the quick brown fox") == pytest.approx(1.0)
    assert _jaccard("fox", "cat") == pytest.approx(0.0)
    assert _jaccard("the fox", "the cat") == pytest.approx(1/3)


def test_judge_skips_no_rationale():
    from scripts.eval.risk_rating import run_llm_judge
    rec = _rec("assignment",
               {"assignment_restricted": True, "consent_required": True,
                "exceptions_detected": [], "restricted_party": "assignor", "confidence": 0.9},
               "high")
    rec.risk_rationale = None  # no rationale → skip
    result = run_llm_judge([rec], llm=MockLLM())
    assert result.n_judged == 0
