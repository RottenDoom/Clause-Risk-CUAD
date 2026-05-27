"""
agent/risk_rating.py  —  Issue 8

Generates the LLM-based risk rating, rationale, and uncertainty notes
for a single clause card. One retry on JSON parse failure; graceful null
degradation on second failure (never crashes).

Interface:
    generate_risk_rating(
        clause_family, extracted_clause_text, structured_interpretation,
        similar_precedents, contrasting_precedents, anthropic_client
    ) -> (RiskLevel | None, str | None, list[str])
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

from config import MAX_TOKENS
from agent.models import ClauseCard, ContrastingPrecedent, RiskLevel, SimilarPrecedent
from services.generation.base import LLMClient


# ---------------------------------------------------------------------------
# Prompt construction helpers
# ---------------------------------------------------------------------------

def _format_similar(similar: list[SimilarPrecedent]) -> str:
    if not similar:
        return "  (none retrieved)"
    lines = []
    for p in similar:
        lines.append(f"- contract_id: {p.contract_id}\n  why_similar: {p.why_similar}")
    return "\n".join(lines)


def _format_contrasting(contrasting: list[ContrastingPrecedent]) -> str:
    if not contrasting:
        return "  (none retrieved)"
    lines = []
    for p in contrasting:
        lines.append(f"- contract_id: {p.contract_id}\n  why_contrasting: {p.why_contrasting}")
    return "\n".join(lines)


def _build_prompt(
    clause_family: str,
    extracted_clause_text: Optional[str],
    structured_interpretation: dict,
    similar: list[SimilarPrecedent],
    contrasting: list[ContrastingPrecedent],
    retry: bool = False,
) -> str:
    clause_block = extracted_clause_text or "Clause not found in contract."
    interp_block = json.dumps(structured_interpretation, indent=2)

    base = (
        f"CLAUSE FAMILY: {clause_family}\n\n"
        f"EXTRACTED CLAUSE TEXT:\n{clause_block}\n\n"
        f"STRUCTURED INTERPRETATION:\n{interp_block}\n\n"
        f"SIMILAR PRECEDENTS (same risk posture as this clause):\n"
        f"{_format_similar(similar)}\n\n"
        f"CONTRASTING PRECEDENTS (different risk posture from this clause):\n"
        f"{_format_contrasting(contrasting)}\n\n"
        "Rate the risk of this clause. Consider:\n"
        "1. What the clause text says and what obligations it creates or restricts.\n"
        "2. What the structured interpretation reveals about restrictions, exceptions, and ambiguities.\n"
        "3. What the similar precedents suggest about how common this posture is.\n"
        "4. What the contrasting precedents reveal about how much riskier or softer this clause is.\n\n"
        "Return JSON with exactly these keys:\n"
        '{"risk_rating": "low" | "medium" | "high", '
        '"risk_rationale": "<2-4 sentence explanation>", '
        '"confidence_uncertainty_notes": ["<note 1>", ...]}'
    )
    if retry:
        base += (
            "\n\nYour previous response was not valid JSON. "
            "Return only the JSON object, no other text."
        )
    return base


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate_risk_rating(
    clause_family: str,
    extracted_clause_text: Optional[str],
    structured_interpretation: dict,
    similar_precedents: list[SimilarPrecedent],
    contrasting_precedents: list[ContrastingPrecedent],
    llm: LLMClient,
) -> tuple[Optional[RiskLevel], Optional[str], list[str]]:
    """
    Returns (risk_rating, risk_rationale, confidence_uncertainty_notes).
    On double parse failure: (None, None, [error_note]).
    """
    system = (
        "You are a commercial contract risk analyst performing due diligence. "
        "Allowed risk levels: low, medium, high. "
        "Respond only in JSON with no markdown fences."
    )

    for attempt in range(2):
        if attempt == 1:
            logger.warning("family=%s risk rating retry (attempt 2)", clause_family)

        prompt = _build_prompt(
            clause_family,
            extracted_clause_text,
            structured_interpretation,
            similar_precedents,
            contrasting_precedents,
            retry=(attempt == 1),
        )
        try:
            parsed = llm.generate_json(prompt, system=system, max_tokens=MAX_TOKENS)

            rating_str = parsed.get("risk_rating", "").lower()
            try:
                risk_level = RiskLevel(rating_str)
            except ValueError:
                risk_level = RiskLevel.medium

            rationale = parsed.get("risk_rationale") or ""
            notes: list[str] = parsed.get("confidence_uncertainty_notes") or []
            if isinstance(notes, str):
                notes = [notes]

            logger.info("family=%s risk=%s attempt=%d", clause_family, risk_level.value, attempt + 1)
            return risk_level, rationale, notes

        except Exception as exc:
            logger.warning("family=%s risk rating parse failure attempt=%d — %s", clause_family, attempt + 1, exc)
            if attempt == 1:
                logger.error("family=%s risk rating failed after 2 attempts — degrading to None", clause_family)
                return (
                    None,
                    None,
                    ["Risk rating could not be generated: LLM response was not parseable JSON."],
                )

    return None, None, ["Unexpected error during risk rating generation."]


def generate_risk_card(
    family: str,
    clause_text: Optional[str],
    clause_found: bool,
    similar: list[SimilarPrecedent],
    contrasting: list[ContrastingPrecedent],
    llm: LLMClient,
    structured_interpretation: Optional[dict] = None,
    discovery_score: float = 0.0,
) -> ClauseCard:
    """
    Convenience wrapper called by loop.py. Runs interpret_clause (if needed)
    then generate_risk_rating, and assembles a ClauseCard.
    """
    from agent.interpretation import get_null_interpretation, interpret_clause

    if clause_found and clause_text:
        interp = structured_interpretation or interpret_clause(clause_text, family, llm)
    else:
        interp = get_null_interpretation(family)

    risk_level, rationale, notes = generate_risk_rating(
        clause_family=family,
        extracted_clause_text=clause_text,
        structured_interpretation=interp,
        similar_precedents=similar,
        contrasting_precedents=contrasting,
        llm=llm,
    )

    return ClauseCard(
        clause_family=family,
        clause_found=clause_found,
        extracted_clause_text=clause_text,
        structured_interpretation=interp,
        similar_precedents=similar,
        contrasting_precedents=contrasting,
        llm_generated_risk_rating=risk_level,
        risk_rationale=rationale,
        confidence_uncertainty_notes=notes,
        discovery_score=discovery_score,
    )
