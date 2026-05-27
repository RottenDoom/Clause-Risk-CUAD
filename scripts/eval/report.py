"""scripts/eval/report.py — stdout table and JSON writer."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path


def _to_dict(obj) -> dict:
    if dataclasses.is_dataclass(obj):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def write_results_json(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {stage: _to_dict(metrics) for stage, metrics in results.items()}
    path.write_text(json.dumps(serializable, indent=2))


_ROWS = [
    # (display_name, stage_key, field_path, target, higher_is_better)
    ("Accuracy",            "clause_discovery", "accuracy",               0.70,  True),
    ("Precision",           "clause_discovery", "precision",              0.60,  True),
    ("Recall",              "clause_discovery", "recall",                 0.65,  True),
    ("F1",                  "clause_discovery", "f1",                     None,  True),
    ("F2",                  "clause_discovery", "f2",                     None,  True),
    ("Laziness rate",       "clause_discovery", "laziness_rate",          None,  False),
    ("P@80% Recall",        "clause_discovery", "precision_at_80_recall", None,  True),
    ("Recall@3",            "retrieval",        "recall_at_3",            None,  True),
    ("MRR",                 "retrieval",        "mrr",                    0.40,  True),
    ("nDCG@3",              "retrieval",        "ndcg_at_3",              0.70,  True),
    ("Avg semantic sim",    "retrieval",        "avg_semantic_sim",       None,  True),
    ("Field completeness",  "interpretation",   "field_completeness_rate",None,  True),
    ("Heuristic agree %",   "risk_rating",      "heuristic.heuristic_agreement_pct", None, True),
]


def _get_nested(obj, path: str):
    """Resolve 'heuristic.heuristic_agreement_pct' style paths."""
    for key in path.split("."):
        if dataclasses.is_dataclass(obj):
            obj = getattr(obj, key, None)
        elif isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return None
        if obj is None:
            return None
    return obj


def print_table(results: dict) -> None:
    header = f"{'Metric':<28} {'Value':>8}  {'Target':>8}  {'Pass':>5}"
    print("\n" + "=" * 58)
    print("  CONTRACT REVIEW PIPELINE — EVALUATION RESULTS")
    print("=" * 58)
    print(header)
    print("-" * 58)

    for display, stage_key, field_path, target, higher_better in _ROWS:
        stage_data = results.get(stage_key)
        if stage_data is None:
            continue
        value = _get_nested(stage_data, field_path)
        if value is None:
            val_str = "    N/A"
            pass_str = "  —"
        else:
            val_str = f"{value:8.3f}"
            if target is not None:
                ok = value >= target if higher_better else value <= target
                pass_str = "  ✓" if ok else "  ✗"
            else:
                pass_str = "  —"
        tgt_str = f"{target:8.2f}" if target is not None else "     N/A"
        print(f"  {display:<26} {val_str}  {tgt_str}  {pass_str}")

    # LLM judge section (--full mode)
    risk = results.get("risk_rating")
    if risk and getattr(risk, "avg_judge_score", None) is not None:
        print("-" * 58)
        print(f"  {'LLM judge score (1-5)':<26} {risk.avg_judge_score:8.2f}       —       —")
        if risk.judge_pass_pct is not None:
            print(f"  {'Judge pass %':<26} {risk.judge_pass_pct:8.3f}       —       —")
        if risk.avg_jaccard_sim is not None:
            print(f"  {'Rationale Jaccard sim':<26} {risk.avg_jaccard_sim:8.3f}       —       —")

    print("=" * 58 + "\n")
