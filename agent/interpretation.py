"""
agent/interpretation.py  —  Issue 6

Two-pass structured interpretation:
  Pass 1 — regex/keyword heuristics for deterministic fields (no LLM)
  Pass 2 — single Claude call for fields that require semantic reasoning

Interface:
    interpret_clause(clause_text, clause_family, anthropic_client) -> dict
    get_null_interpretation(clause_family) -> dict
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    INTERPRETATION_CONFIDENCE_PENALTY_PER_LLM_FIELD,
    INTERPRETATION_MIN_WORDS,
    INTERPRETATION_SHORT_TEXT_PENALTY,
    MAX_TOKENS,
    MODEL,
)


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

def _find_day_count(text: str, *keywords: str) -> Optional[int]:
    """
    Find 'N days' (or '(N) days') in any sentence that contains one of the keywords.
    Splits on sentence boundaries so "thirty (30) days prior written notice" is found
    by keyword "notice" even with intervening words.
    """
    segments = re.split(r"[.;]", text)
    for seg in segments:
        if not any(re.search(kw, seg, re.IGNORECASE) for kw in keywords):
            continue
        m = re.search(r"\(?\b(\d+)\)?\s*days?\b", seg, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _contains(text: str, *keywords: str) -> bool:
    low = text.lower()
    return any(k in low for k in keywords)


def _find_exceptions(text: str) -> list[str]:
    """Scan for common assignment/change-of-control exception keywords."""
    found = []
    kw_map = {
        "affiliate": "affiliate",
        "subsidiary": "subsidiary",
        "merger": "merger",
        "acquisition": "acquisition",
        "successor": "successor",
        "change of control": "change_of_control",
        "reorganization": "reorganization",
        "ipo": "IPO",
    }
    low = text.lower()
    for kw, label in kw_map.items():
        if kw in low:
            found.append(label)
    return found


# ---------------------------------------------------------------------------
# Heuristic passes (Pass 1) — family-specific
# ---------------------------------------------------------------------------

def _heuristic_assignment(text: str) -> tuple[dict, int]:
    """Returns (partial_fields, n_llm_fields_needed)."""
    low = text.lower()
    assignment_restricted = _contains(
        low, "shall not assign", "may not assign", "cannot assign",
        "without consent", "without prior written consent", "not be assigned",
    )
    consent_required = _contains(
        low, "written consent", "prior written consent", "prior approval",
        "written approval",
    )
    exceptions = _find_exceptions(text)
    return {
        "assignment_restricted": assignment_restricted,
        "consent_required": consent_required,
        "exceptions_detected": exceptions,
        # restricted_party requires LLM
    }, 1


def _heuristic_change_of_control(text: str) -> tuple[dict, int]:
    low = text.lower()
    trigger = _contains(
        low, "change of control", "change in control", "merger", "acquisition",
        "change in ownership",
    )
    termination_right = _contains(low, "terminat", "may terminate", "right to terminate")
    consent_required = _contains(low, "written consent", "prior approval", "consent")
    exceptions = _find_exceptions(text)
    return {
        "trigger_on_control_change": trigger,
        "termination_right": termination_right,
        "consent_required": consent_required,
        "exceptions_detected": exceptions,
        # acquirer_bound requires LLM
    }, 1


def _heuristic_termination(text: str) -> tuple[dict, int]:
    low = text.lower()
    convenience = _contains(
        low, "for convenience", "without cause", "for any reason",
        "at any time", "for no reason",
    )
    mutual = (
        _contains(low, "either party") and _contains(low, "terminat")
    )
    grounds: list[str] = []
    for kw in ("breach", "convenience", "insolvency", "bankruptcy", "default",
               "material breach", "cause"):
        if kw in low:
            grounds.append(kw.replace(" ", "_"))
    notice_days = _find_day_count(text, "notice")
    cure_days = _find_day_count(text, "cure")
    return {
        "termination_for_convenience": convenience,
        "notice_period_days": notice_days,
        "cure_period_days": cure_days,
        "mutual_termination_right": mutual,
        "grounds_detected": grounds,
    }, 0  # all fields determined heuristically


def _heuristic_exclusivity(text: str) -> tuple[dict, int]:
    low = text.lower()
    exclusive = _contains(
        low, "exclusiv", "sole provider", "exclusively", "only provider",
        "shall not engage", "shall not purchase",
    )
    non_compete = _contains(low, "non-compete", "non compete", "noncompete", "competing")
    geographic = _contains(low, "geographic", "territory", "region", "area", "worldwide")
    duration = bool(re.search(r"\d+\s*(year|month|day)", low))
    return {
        "exclusivity_granted": exclusive,
        "non_compete_present": non_compete,
        "geographic_limitation": geographic,
        "duration_mentioned": duration,
        # scope requires LLM
    }, 1


# ---------------------------------------------------------------------------
# Claude pass (Pass 2) — only for fields that need semantic reasoning
# ---------------------------------------------------------------------------

_SEMANTIC_PROMPTS: dict[str, str] = {
    "assignment": (
        'Return JSON with one key "restricted_party" '
        'whose value is exactly one of: "assignor", "assignee", "both", "neither", "unknown". '
        "The value describes which contracting party faces assignment restrictions."
    ),
    "change_of_control": (
        'Return JSON with one key "acquirer_bound" '
        'whose value is a boolean. True if the clause states or implies that '
        "a new controlling party or acquirer becomes bound by the agreement."
    ),
    "exclusivity": (
        'Return JSON with one key "scope" '
        'whose value is exactly one of: "full", "partial", "none", "unknown". '
        '"full" means complete exclusivity across the whole market/service; '
        '"partial" means limited by geography, product line, or customer segment.'
    ),
}


def _call_claude_for_semantic_field(
    clause_text: str,
    clause_family: str,
    anthropic_client,
) -> dict:
    """One targeted Claude call to resolve semantic fields for the given family."""
    field_spec = _SEMANTIC_PROMPTS.get(clause_family, "")
    if not field_spec:
        return {}

    prompt = (
        f"CLAUSE FAMILY: {clause_family}\n\n"
        f"CLAUSE TEXT:\n{clause_text}\n\n"
        f"{field_spec}"
    )
    try:
        response = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=128,
            system=(
                "You are a legal clause analyzer. "
                "Respond only in JSON with no markdown fences."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Confidence computation
# ---------------------------------------------------------------------------

def _compute_confidence(
    clause_text: str,
    n_llm_fields: int,
) -> float:
    word_count = len(clause_text.split())
    score = 1.0
    score -= n_llm_fields * INTERPRETATION_CONFIDENCE_PENALTY_PER_LLM_FIELD
    if word_count < INTERPRETATION_MIN_WORDS:
        score -= INTERPRETATION_SHORT_TEXT_PENALTY
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def interpret_clause(
    clause_text: str,
    clause_family: str,
    anthropic_client,
) -> dict:
    """
    Two-pass structured interpretation.
    Returns a dict conforming to the appropriate per-family Pydantic model.
    """
    heuristic_fn = {
        "assignment": _heuristic_assignment,
        "change_of_control": _heuristic_change_of_control,
        "termination": _heuristic_termination,
        "exclusivity": _heuristic_exclusivity,
    }.get(clause_family)

    if heuristic_fn is None:
        raise ValueError(f"Unknown clause family: {clause_family!r}")

    fields, n_llm_fields = heuristic_fn(clause_text)

    if n_llm_fields > 0:
        semantic = _call_claude_for_semantic_field(clause_text, clause_family, anthropic_client)
        fields.update(semantic)

    # Fill in any missing semantic fields with safe defaults
    _apply_defaults(fields, clause_family)

    fields["confidence"] = _compute_confidence(clause_text, n_llm_fields)
    return fields


def _apply_defaults(fields: dict, family: str) -> None:
    """Fill semantic fields with 'unknown'/False if Claude returned nothing."""
    if family == "assignment":
        fields.setdefault("restricted_party", "unknown")
    elif family == "change_of_control":
        fields.setdefault("acquirer_bound", False)
    elif family == "exclusivity":
        fields.setdefault("scope", "unknown")


def get_null_interpretation(clause_family: str) -> dict:
    """
    Returns a safe all-null/False interpretation dict for when clause_found=False.
    Matches the per-family model shape so serialization never fails.
    """
    null_map: dict[str, dict] = {
        "assignment": {
            "assignment_restricted": False,
            "consent_required": False,
            "restricted_party": "unknown",
            "exceptions_detected": [],
            "confidence": 0.0,
        },
        "change_of_control": {
            "trigger_on_control_change": False,
            "termination_right": False,
            "consent_required": False,
            "acquirer_bound": False,
            "exceptions_detected": [],
            "confidence": 0.0,
        },
        "termination": {
            "termination_for_convenience": False,
            "notice_period_days": None,
            "cure_period_days": None,
            "mutual_termination_right": False,
            "grounds_detected": [],
            "confidence": 0.0,
        },
        "exclusivity": {
            "exclusivity_granted": False,
            "scope": "none",
            "geographic_limitation": False,
            "non_compete_present": False,
            "duration_mentioned": False,
            "confidence": 0.0,
        },
    }
    if clause_family not in null_map:
        raise ValueError(f"Unknown clause family: {clause_family!r}")
    return null_map[clause_family]
