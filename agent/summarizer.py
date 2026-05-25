"""
agent/summarizer.py  —  Issue 9

Rule-based aggregate risk (no LLM) + single Claude call for overall_summary
and top_red_flags.

Interface:
    summarize_contract(contract_id, clause_cards, anthropic_client)
        -> (overall_summary: str, overall_risk_rating: RiskLevel, top_red_flags: list[str])
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MAX_TOKENS, MODEL
from agent.models import ClauseCard, RiskLevel


# ---------------------------------------------------------------------------
# Rule-based aggregate risk
# ---------------------------------------------------------------------------

def _aggregate_risk(cards: List[ClauseCard]) -> tuple[RiskLevel, list[str]]:
    """
    Max-based aggregation:
      any high  → high
      any medium (or null, treated as medium) → medium
      else      → low

    Returns (overall_risk, extra_red_flag_notes).
    Null ratings produce a red-flag note so reviewers know a card was skipped.
    """
    extra_flags: list[str] = []
    effective_levels: list[RiskLevel] = []

    for card in cards:
        if card.llm_generated_risk_rating is None:
            effective_levels.append(RiskLevel.medium)
            extra_flags.append(
                f"{card.clause_family.replace('_', ' ').title()}: risk rating generation "
                "failed — treated as medium for aggregate calculation."
            )
        else:
            effective_levels.append(card.llm_generated_risk_rating)

    if RiskLevel.high in effective_levels:
        return RiskLevel.high, extra_flags
    if RiskLevel.medium in effective_levels:
        return RiskLevel.medium, extra_flags
    return RiskLevel.low, extra_flags


# ---------------------------------------------------------------------------
# Claude call for summary + red flags
# ---------------------------------------------------------------------------

def _format_cards_for_prompt(cards: List[ClauseCard]) -> str:
    parts = []
    for card in cards:
        rating = card.llm_generated_risk_rating.value if card.llm_generated_risk_rating else "null"
        found_str = "Found" if card.clause_found else "Not found"
        rationale = card.risk_rationale or "N/A"
        notes = "; ".join(card.confidence_uncertainty_notes) or "None"
        parts.append(
            f"[{card.clause_family.upper()}] ({found_str}) — Risk: {rating}\n"
            f"  Rationale: {rationale}\n"
            f"  Uncertainty: {notes}"
        )
    return "\n\n".join(parts)


def _claude_summary(
    contract_id: str,
    cards: List[ClauseCard],
    anthropic_client,
) -> tuple[str, list[str]]:
    cards_block = _format_cards_for_prompt(cards)
    prompt = (
        f"CONTRACT: {contract_id}\n\n"
        f"CLAUSE RISK CARDS:\n{cards_block}\n\n"
        "Based on these four clause-family risk cards, produce:\n"
        "1. A 3-5 sentence plain English overall risk summary for a commercial reviewer.\n"
        "2. A list of the top red flags or areas requiring human review "
        "(drawn from the risk rationales and uncertainty notes).\n\n"
        "Return JSON:\n"
        '{"overall_summary": "...", "top_red_flags": ["...", ...]}'
    )
    try:
        resp = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=(
                "You are a commercial contract risk analyst. "
                "Respond only in JSON with no markdown fences."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        summary = parsed.get("overall_summary", "")
        red_flags: list[str] = parsed.get("top_red_flags") or []
        if isinstance(red_flags, str):
            red_flags = [red_flags]
        return summary, red_flags
    except Exception:
        # Graceful degradation: synthesize from card rationales
        summary = (
            f"Contract {contract_id} reviewed across four clause families. "
            "See individual clause cards for details."
        )
        flags = [
            f"{c.clause_family}: {c.risk_rationale or 'review required'}"
            for c in cards
            if c.llm_generated_risk_rating in (RiskLevel.high, RiskLevel.medium, None)
        ]
        return summary, flags or ["Manual review recommended."]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def summarize_contract(
    contract_id: str,
    clause_cards: List[ClauseCard],
    anthropic_client,
) -> tuple[str, RiskLevel, list[str]]:
    """
    Returns (overall_summary, overall_risk_rating, top_red_flags).
    """
    overall_risk, null_flags = _aggregate_risk(clause_cards)
    overall_summary, llm_flags = _claude_summary(contract_id, clause_cards, anthropic_client)
    top_red_flags = llm_flags + null_flags
    return overall_summary, overall_risk, top_red_flags
