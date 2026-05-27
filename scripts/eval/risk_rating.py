"""scripts/eval/risk_rating.py — Stage 4 metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from scripts.eval.loader import EvalRecord


# ---------------------------------------------------------------------------
# Heuristic risk labeling — derived from structured_interpretation fields.
# Rules mirror real-world legal risk conventions for each family.
# ---------------------------------------------------------------------------

def heuristic_risk_label(family: str, interp: dict) -> str:
    """Return 'high', 'medium', or 'low' based on deterministic field rules."""
    if family == "assignment":
        if interp.get("consent_required") and not interp.get("exceptions_detected"):
            return "high"
        if interp.get("consent_required"):
            return "medium"
        return "low"

    if family == "change_of_control":
        if interp.get("termination_right") and not interp.get("exceptions_detected"):
            return "high"
        if interp.get("trigger_on_control_change"):
            return "medium"
        return "low"

    if family == "termination":
        notice = interp.get("notice_period_days")
        if interp.get("termination_for_convenience") and (notice is None or notice <= 30):
            return "high"
        if interp.get("termination_for_convenience"):
            return "medium"
        return "low"

    if family == "exclusivity":
        if interp.get("exclusivity_granted") and interp.get("non_compete_present"):
            return "high"
        if interp.get("exclusivity_granted"):
            return "medium"
        return "low"

    return "medium"  # unknown family — conservative default


@dataclass
class HeuristicMetrics:
    heuristic_agreement_pct: float
    n_compared: int


@dataclass
class RiskMetrics:
    heuristic: HeuristicMetrics
    # Populated only in --full mode by run_llm_judge()
    avg_judge_score: Optional[float] = None
    avg_jaccard_sim: Optional[float] = None
    judge_pass_pct: Optional[float] = None
    n_judged: int = 0


def compute_heuristic_metrics(records: list[EvalRecord]) -> HeuristicMetrics:
    evaluated = [
        r for r in records
        if r.clause_found
        and r.structured_interpretation is not None
        and r.llm_generated_risk_rating is not None
    ]
    if not evaluated:
        return HeuristicMetrics(0.0, 0)

    agreements = sum(
        1 for r in evaluated
        if heuristic_risk_label(r.family, r.structured_interpretation)
        == r.llm_generated_risk_rating
    )
    return HeuristicMetrics(
        heuristic_agreement_pct=agreements / len(evaluated),
        n_compared=len(evaluated),
    )


# ---------------------------------------------------------------------------
# LLM-as-judge (--full mode)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are evaluating a contract risk card produced by an automated pipeline. "
    "Score the rationale quality 1-5: "
    "5=accurate, specific, cites clause language; "
    "4=mostly accurate with minor gaps; "
    "3=partially correct or vague; "
    "2=largely incorrect or generic; "
    "1=completely wrong or empty. "
    "Also judge whether the risk rating (low/medium/high) is plausible. "
    "Respond only in JSON: "
    '{"score": <1-5>, "rating_plausible": <true/false>, "reference_rationale": "<1-2 sentences>"}'
)


def run_llm_judge(records: list[EvalRecord], llm) -> RiskMetrics:
    """
    Call LLM once per record (clause_found=True, rationale present).
    Returns RiskMetrics with judge fields populated.
    """
    to_judge = [
        r for r in records
        if r.clause_found and r.risk_rationale and r.llm_generated_risk_rating
    ]
    if not to_judge:
        return RiskMetrics(heuristic=compute_heuristic_metrics(records))

    scores, plausible, jaccards = [], [], []
    for rec in to_judge:
        prompt = (
            f"CLAUSE FAMILY: {rec.family}\n\n"
            f"EXTRACTED CLAUSE:\n{rec.extracted_clause_text or '(not found)'}\n\n"
            f"PIPELINE RISK RATING: {rec.llm_generated_risk_rating}\n"
            f"PIPELINE RATIONALE: {rec.risk_rationale}\n\n"
            "Evaluate the above risk card."
        )
        try:
            parsed = llm.generate_json(prompt, system=_JUDGE_SYSTEM, max_tokens=256)
            score = int(parsed.get("score", 3))
            is_plausible = bool(parsed.get("rating_plausible", True))
            ref_rationale = parsed.get("reference_rationale", "")
            scores.append(score)
            plausible.append(1.0 if is_plausible else 0.0)
            jaccards.append(_jaccard(rec.risk_rationale, ref_rationale))
        except Exception:
            pass  # failed judge call → skip this record

    heuristic = compute_heuristic_metrics(records)
    return RiskMetrics(
        heuristic=heuristic,
        avg_judge_score=sum(scores) / len(scores) if scores else None,
        avg_jaccard_sim=sum(jaccards) / len(jaccards) if jaccards else None,
        judge_pass_pct=sum(plausible) / len(plausible) if plausible else None,
        n_judged=len(scores),
    )


def _jaccard(text_a: str, text_b: str) -> float:
    words_a = set((text_a or "").lower().split())
    words_b = set((text_b or "").lower().split())
    if not words_a and not words_b:
        return 1.0
    union = words_a | words_b
    return len(words_a & words_b) / len(union) if union else 0.0
