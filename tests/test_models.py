"""
tests/test_models.py

Unit tests for agent/models.py — verify all Pydantic schemas instantiate
and serialize to valid JSON with expected shapes.
"""

import json
import pytest
from agent.models import (
    AssignmentInterpretation,
    ChangeOfControlInterpretation,
    ClauseCard,
    ContractReviewOutput,
    ContrastingPrecedent,
    ExclusivityInterpretation,
    RiskLevel,
    SimilarPrecedent,
    TerminationInterpretation,
)


def test_risk_level_values():
    assert RiskLevel.low.value == "low"
    assert RiskLevel.medium.value == "medium"
    assert RiskLevel.high.value == "high"
    assert RiskLevel("low") == RiskLevel.low


def test_similar_precedent():
    p = SimilarPrecedent(contract_id="ref_001", why_similar="Both restrict assignment without consent.")
    assert p.contract_id == "ref_001"
    assert "restrict" in p.why_similar


def test_contrasting_precedent():
    p = ContrastingPrecedent(contract_id="ref_002", why_contrasting="Reference allows M&A exception.")
    d = p.model_dump()
    assert d["contract_id"] == "ref_002"
    assert d["why_contrasting"]


def test_assignment_interpretation():
    obj = AssignmentInterpretation(
        assignment_restricted=True,
        consent_required=True,
        restricted_party="both",
        exceptions_detected=["affiliate", "merger"],
        confidence=0.85,
    )
    assert obj.assignment_restricted is True
    assert 0.0 <= obj.confidence <= 1.0


def test_change_of_control_interpretation():
    obj = ChangeOfControlInterpretation(
        trigger_on_control_change=True,
        termination_right=True,
        consent_required=False,
        acquirer_bound=True,
        exceptions_detected=[],
        confidence=0.7,
    )
    assert obj.trigger_on_control_change is True


def test_termination_interpretation():
    obj = TerminationInterpretation(
        termination_for_convenience=True,
        notice_period_days=30,
        cure_period_days=15,
        mutual_termination_right=True,
        grounds_detected=["breach", "convenience"],
        confidence=0.9,
    )
    assert obj.notice_period_days == 30
    assert obj.cure_period_days == 15


def test_termination_interpretation_nulls():
    obj = TerminationInterpretation(
        termination_for_convenience=False,
        notice_period_days=None,
        cure_period_days=None,
        mutual_termination_right=False,
        grounds_detected=[],
        confidence=0.0,
    )
    assert obj.notice_period_days is None


def test_exclusivity_interpretation():
    obj = ExclusivityInterpretation(
        exclusivity_granted=True,
        scope="full",
        geographic_limitation=False,
        non_compete_present=True,
        duration_mentioned=True,
        confidence=0.75,
    )
    assert obj.scope == "full"


def _make_clause_card(family: str, found: bool = True) -> ClauseCard:
    return ClauseCard(
        clause_family=family,
        clause_found=found,
        extracted_clause_text="No party shall assign..." if found else None,
        structured_interpretation={"assignment_restricted": True, "confidence": 0.8} if found else None,
        similar_precedents=[SimilarPrecedent(contract_id="ref_001", why_similar="Similar restriction.")] if found else [],
        contrasting_precedents=[ContrastingPrecedent(contract_id="ref_002", why_contrasting="Allows M&A.")] if found else [],
        llm_generated_risk_rating=RiskLevel.high if found else None,
        risk_rationale="High risk due to no exceptions." if found else None,
        confidence_uncertainty_notes=["No M&A carve-out found."],
    )


def test_clause_card_found():
    card = _make_clause_card("assignment", found=True)
    assert card.clause_found is True
    assert card.llm_generated_risk_rating == RiskLevel.high
    assert len(card.similar_precedents) == 1


def test_clause_card_not_found():
    card = _make_clause_card("termination", found=False)
    assert card.clause_found is False
    assert card.extracted_clause_text is None
    assert card.similar_precedents == []


def test_clause_card_has_discovery_score():
    card = ClauseCard(
        clause_family="assignment", clause_found=True,
        extracted_clause_text="text", structured_interpretation={},
        similar_precedents=[], contrasting_precedents=[],
        llm_generated_risk_rating=None, risk_rationale=None,
        confidence_uncertainty_notes=[],
    )
    assert hasattr(card, "discovery_score")
    assert card.discovery_score == 0.0


def test_clause_card_discovery_score_stored():
    card = ClauseCard(
        clause_family="termination", clause_found=True,
        extracted_clause_text="text", structured_interpretation={},
        similar_precedents=[], contrasting_precedents=[],
        llm_generated_risk_rating=None, risk_rationale=None,
        confidence_uncertainty_notes=[], discovery_score=0.72,
    )
    assert card.discovery_score == 0.72


def test_contract_review_output_serializes():
    cards = [
        _make_clause_card("assignment"),
        _make_clause_card("change_of_control", found=False),
        _make_clause_card("termination"),
        _make_clause_card("exclusivity", found=False),
    ]
    output = ContractReviewOutput(
        contract_id="TestContract",
        overall_summary="This contract has high risk in the assignment clause.",
        overall_risk_rating=RiskLevel.high,
        top_red_flags=["No M&A exception in assignment clause."],
        clause_cards=cards,
    )
    assert output.contract_id == "TestContract"
    assert len(output.clause_cards) == 4

    # Round-trip through JSON
    raw = output.model_dump_json()
    parsed = json.loads(raw)
    assert parsed["overall_risk_rating"] == "high"
    assert len(parsed["clause_cards"]) == 4
