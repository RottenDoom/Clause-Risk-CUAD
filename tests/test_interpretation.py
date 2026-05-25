"""
tests/test_interpretation.py

Tests for agent/interpretation.py — heuristic pass only (no Claude calls).
Claude-dependent paths are skipped or mocked.
"""

import pytest
from unittest.mock import MagicMock, patch

from agent.interpretation import (
    get_null_interpretation,
    interpret_clause,
    _heuristic_assignment,
    _heuristic_change_of_control,
    _heuristic_termination,
    _heuristic_exclusivity,
    _find_day_count,
)


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------

def test_find_day_count_before():
    assert _find_day_count("30 days notice required", "notice") == 30


def test_find_day_count_parens():
    assert _find_day_count("upon thirty (30) days prior written notice", "notice") == 30


def test_find_day_count_none():
    assert _find_day_count("no time specified", "notice") is None


# ---------------------------------------------------------------------------
# Heuristic pass — assignment
# ---------------------------------------------------------------------------

ASSIGN_TEXT = (
    "Neither party shall assign this agreement or any of its rights or "
    "obligations hereunder without the prior written consent of the other party. "
    "Notwithstanding the foregoing, either party may assign this agreement to an "
    "affiliate or subsidiary without consent."
)

def test_heuristic_assignment_restricted():
    fields, _ = _heuristic_assignment(ASSIGN_TEXT)
    assert fields["assignment_restricted"] is True
    assert fields["consent_required"] is True
    assert "affiliate" in fields["exceptions_detected"]
    assert "subsidiary" in fields["exceptions_detected"]


def test_heuristic_assignment_free():
    text = "Either party may freely assign all rights under this agreement."
    fields, _ = _heuristic_assignment(text)
    assert fields["assignment_restricted"] is False
    assert fields["consent_required"] is False


# ---------------------------------------------------------------------------
# Heuristic pass — change_of_control
# ---------------------------------------------------------------------------

COC_TEXT = (
    "In the event of a change of control of either party, the other party "
    "shall have the right to terminate this agreement upon thirty (30) days "
    "written notice."
)

def test_heuristic_coc_trigger():
    fields, _ = _heuristic_change_of_control(COC_TEXT)
    assert fields["trigger_on_control_change"] is True
    assert fields["termination_right"] is True


# ---------------------------------------------------------------------------
# Heuristic pass — termination
# ---------------------------------------------------------------------------

TERM_TEXT = (
    "Either party may terminate this agreement for convenience upon thirty (30) "
    "days prior written notice. In the event of a material breach, the non-breaching "
    "party may terminate after a fifteen (15) day cure period."
)

def test_heuristic_termination_convenience():
    fields, n_llm = _heuristic_termination(TERM_TEXT)
    assert fields["termination_for_convenience"] is True
    assert fields["mutual_termination_right"] is True
    assert fields["notice_period_days"] == 30
    assert fields["cure_period_days"] == 15
    assert n_llm == 0  # all fields resolved heuristically


def test_heuristic_termination_grounds():
    fields, _ = _heuristic_termination(TERM_TEXT)
    assert "breach" in fields["grounds_detected"] or "material_breach" in fields["grounds_detected"]
    assert "convenience" in fields["grounds_detected"]


# ---------------------------------------------------------------------------
# Heuristic pass — exclusivity
# ---------------------------------------------------------------------------

EXCL_TEXT = (
    "During the term of this agreement, Customer shall exclusively purchase "
    "the Services from Vendor and shall not purchase competing services from "
    "any third party within the defined geographic territory."
)

def test_heuristic_exclusivity():
    fields, _ = _heuristic_exclusivity(EXCL_TEXT)
    assert fields["exclusivity_granted"] is True
    assert fields["geographic_limitation"] is True
    assert fields["non_compete_present"] is True


# ---------------------------------------------------------------------------
# get_null_interpretation
# ---------------------------------------------------------------------------

def test_null_interpretation_all_families():
    for family in ["assignment", "change_of_control", "termination", "exclusivity"]:
        null = get_null_interpretation(family)
        assert isinstance(null, dict)
        assert null["confidence"] == 0.0


def test_null_interpretation_assignment_shape():
    null = get_null_interpretation("assignment")
    assert null["assignment_restricted"] is False
    assert null["restricted_party"] == "unknown"
    assert null["exceptions_detected"] == []


def test_null_interpretation_unknown_family():
    with pytest.raises(ValueError):
        get_null_interpretation("unknown_family")


# ---------------------------------------------------------------------------
# interpret_clause — mocked Claude call
# ---------------------------------------------------------------------------

def _make_mock_client(response_text: str):
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=response_text)]
    client.messages.create.return_value = msg
    return client


def test_interpret_assignment_with_mock():
    client = _make_mock_client('{"restricted_party": "both"}')
    result = interpret_clause(ASSIGN_TEXT, "assignment", client)
    assert result["assignment_restricted"] is True
    assert result["restricted_party"] == "both"
    assert 0.0 <= result["confidence"] <= 1.0


def test_interpret_termination_no_llm_call():
    client = _make_mock_client("")
    result = interpret_clause(TERM_TEXT, "termination", client)
    # termination has n_llm_fields=0, so client should NOT be called
    client.messages.create.assert_not_called()
    assert result["notice_period_days"] == 30


def test_interpret_exclusivity_with_mock():
    client = _make_mock_client('{"scope": "partial"}')
    result = interpret_clause(EXCL_TEXT, "exclusivity", client)
    assert result["scope"] == "partial"
    assert result["exclusivity_granted"] is True


def test_interpret_unknown_family():
    client = _make_mock_client("")
    with pytest.raises(ValueError):
        interpret_clause("some text", "unknown", client)


def test_interpret_claude_failure_uses_defaults():
    client = MagicMock()
    client.messages.create.side_effect = Exception("API error")
    # Should not raise; falls back to default "unknown"
    result = interpret_clause(ASSIGN_TEXT, "assignment", client)
    assert result["restricted_party"] == "unknown"
    assert 0.0 <= result["confidence"] <= 1.0
