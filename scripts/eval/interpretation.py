"""scripts/eval/interpretation.py — Stage 3 field completeness."""
from __future__ import annotations

from dataclasses import dataclass, field

from scripts.eval.loader import EvalRecord

# Expected fields per family, derived from agent/models.py interpretation models.
EXPECTED_FIELDS: dict[str, list[str]] = {
    "assignment": [
        "assignment_restricted", "consent_required",
        "restricted_party", "exceptions_detected", "confidence",
    ],
    "change_of_control": [
        "trigger_on_control_change", "termination_right",
        "consent_required", "acquirer_bound",
        "exceptions_detected", "confidence",
    ],
    "termination": [
        "termination_for_convenience", "notice_period_days",
        "cure_period_days", "mutual_termination_right",
        "grounds_detected", "confidence",
    ],
    "exclusivity": [
        "exclusivity_granted", "scope", "geographic_limitation",
        "non_compete_present", "duration_mentioned", "confidence",
    ],
}


@dataclass
class InterpretationMetrics:
    field_completeness_rate: float
    per_family_completeness: dict[str, float] = field(default_factory=dict)
    n_evaluated: int = 0


def compute_metrics(records: list[EvalRecord]) -> InterpretationMetrics:
    evaluated = [r for r in records if r.clause_found and r.structured_interpretation is not None]
    if not evaluated:
        return InterpretationMetrics(0.0, {}, 0)

    all_scores: list[float] = []
    per_family: dict[str, list[float]] = {}

    for rec in evaluated:
        expected = EXPECTED_FIELDS.get(rec.family, [])
        if not expected:
            continue
        interp = rec.structured_interpretation or {}
        present = sum(1 for f in expected if f in interp)
        score = present / len(expected)
        all_scores.append(score)
        per_family.setdefault(rec.family, []).append(score)

    return InterpretationMetrics(
        field_completeness_rate=sum(all_scores) / len(all_scores) if all_scores else 0.0,
        per_family_completeness={
            fam: sum(scores) / len(scores)
            for fam, scores in per_family.items()
        },
        n_evaluated=len(evaluated),
    )
