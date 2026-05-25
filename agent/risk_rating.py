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
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MAX_TOKENS, MODEL
from agent.models import ContrastingPrecedent, RiskLevel, SimilarPrecedent


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
    anthropic_client,
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
        prompt = _build_prompt(
            clause_family,
            extracted_clause_text,
            structured_interpretation,
            similar_precedents,
            contrasting_precedents,
            retry=(attempt == 1),
        )
        try:
            response = anthropic_client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            clean = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)

            rating_str = parsed.get("risk_rating", "").lower()
            try:
                risk_level = RiskLevel(rating_str)
            except ValueError:
                risk_level = RiskLevel.medium  # safe fallback within the loop

            rationale = parsed.get("risk_rationale") or ""
            notes: list[str] = parsed.get("confidence_uncertainty_notes") or []
            if isinstance(notes, str):
                notes = [notes]

            return risk_level, rationale, notes

        except (json.JSONDecodeError, KeyError, Exception):
            if attempt == 1:
                return (
                    None,
                    None,
                    ["Risk rating could not be generated: LLM response was not parseable JSON."],
                )

    # Should never reach here
    return None, None, ["Unexpected error during risk rating generation."]
