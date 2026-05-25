"""
tests/test_risk_rating.py

Tests for agent/risk_rating.py — Claude calls are mocked.
"""

import json
import pytest
from unittest.mock import MagicMock

from agent.models import ContrastingPrecedent, RiskLevel, SimilarPrecedent
from agent.risk_rating import generate_risk_rating, _build_prompt


def _make_client(response_json: dict | None, fail_first: bool = False):
    client = MagicMock()
    msg = MagicMock()

    if fail_first:
        good_msg = MagicMock()
        good_msg.content = [MagicMock(text=json.dumps(response_json))]
        client.messages.create.side_effect = [Exception("parse error"), good_msg]
    elif response_json is None:
        client.messages.create.side_effect = Exception("always fails")
    else:
        msg.content = [MagicMock(text=json.dumps(response_json))]
        client.messages.create.return_value = msg

    return client


SIMILAR = [SimilarPrecedent(contract_id="ref_001", why_similar="Same restriction.")]
CONTRASTING = [ContrastingPrecedent(contract_id="ref_002", why_contrasting="Allows M&A.")]
INTERP = {"assignment_restricted": True, "consent_required": True, "confidence": 0.8}


def test_generate_risk_rating_success():
    payload = {
        "risk_rating": "high",
        "risk_rationale": "This clause is very restrictive.",
        "confidence_uncertainty_notes": ["No M&A exception."],
    }
    client = _make_client(payload)
    rating, rationale, notes = generate_risk_rating(
        "assignment", "No party shall assign...", INTERP, SIMILAR, CONTRASTING, client
    )
    assert rating == RiskLevel.high
    assert "restrictive" in rationale
    assert len(notes) == 1


def test_generate_risk_rating_medium():
    payload = {
        "risk_rating": "medium",
        "risk_rationale": "Moderate risk.",
        "confidence_uncertainty_notes": [],
    }
    client = _make_client(payload)
    rating, rationale, notes = generate_risk_rating(
        "termination", "Either party may terminate...", INTERP, [], [], client
    )
    assert rating == RiskLevel.medium
    assert notes == []


def test_generate_risk_rating_no_clause():
    payload = {
        "risk_rating": "medium",
        "risk_rationale": "Clause not found; absence may be a risk.",
        "confidence_uncertainty_notes": ["Clause absent."],
    }
    client = _make_client(payload)
    rating, rationale, notes = generate_risk_rating(
        "exclusivity", None, {}, [], [], client
    )
    assert rating is not None
    # Prompt should mention clause not found
    prompt = _build_prompt("exclusivity", None, {}, [], [])
    assert "Clause not found" in prompt


def test_generate_risk_rating_double_failure():
    client = MagicMock()
    client.messages.create.side_effect = Exception("always fails")
    rating, rationale, notes = generate_risk_rating(
        "assignment", "text", INTERP, SIMILAR, CONTRASTING, client
    )
    assert rating is None
    assert rationale is None
    assert len(notes) == 1
    assert "not parseable" in notes[0]


def test_build_prompt_contains_all_sections():
    prompt = _build_prompt("assignment", "No assign clause.", INTERP, SIMILAR, CONTRASTING)
    assert "CLAUSE FAMILY: assignment" in prompt
    assert "No assign clause." in prompt
    assert "SIMILAR PRECEDENTS" in prompt
    assert "CONTRASTING PRECEDENTS" in prompt
    assert "ref_001" in prompt
    assert "ref_002" in prompt


def test_build_prompt_retry_message():
    prompt = _build_prompt("assignment", "text", {}, [], [], retry=True)
    assert "not valid JSON" in prompt


def test_generate_with_invalid_risk_level():
    payload = {
        "risk_rating": "very_high",  # invalid
        "risk_rationale": "Some rationale.",
        "confidence_uncertainty_notes": [],
    }
    client = _make_client(payload)
    rating, rationale, notes = generate_risk_rating(
        "assignment", "text", INTERP, [], [], client
    )
    # Falls back to medium for invalid enum value
    assert rating == RiskLevel.medium
