import json
import pytest
from pathlib import Path


def _make_results():
    from scripts.eval.clause_discovery import DiscoveryMetrics
    from scripts.eval.retrieval import RetrievalMetrics
    from scripts.eval.interpretation import InterpretationMetrics
    from scripts.eval.risk_rating import RiskMetrics, HeuristicMetrics
    return {
        "clause_discovery": DiscoveryMetrics(
            accuracy=0.75, precision=0.80, recall=0.70,
            f1=0.747, f2=0.720, laziness_rate=0.30,
            precision_at_80_recall=None, tp=14, fp=4, fn=6, tn=16, n_records=40,
        ),
        "retrieval": RetrievalMetrics(
            recall_at_3=0.65, mrr=0.58, ndcg_at_3=0.62,
            avg_semantic_sim=0.71, n_queries=14,
        ),
        "interpretation": InterpretationMetrics(
            field_completeness_rate=0.88,
            per_family_completeness={"assignment": 0.90, "termination": 0.85},
            n_evaluated=14,
        ),
        "risk_rating": RiskMetrics(
            heuristic=HeuristicMetrics(heuristic_agreement_pct=0.71, n_compared=14),
        ),
    }


def test_write_json(tmp_path):
    from scripts.eval.report import write_results_json
    results = _make_results()
    out_path = tmp_path / "results.json"
    write_results_json(results, out_path)
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["clause_discovery"]["accuracy"] == pytest.approx(0.75)
    assert data["retrieval"]["mrr"] == pytest.approx(0.58)


def test_format_table_contains_metrics(capsys):
    from scripts.eval.report import print_table
    results = _make_results()
    print_table(results)
    captured = capsys.readouterr()
    assert "accuracy" in captured.out.lower()
    assert "recall@3" in captured.out.lower()
    assert "0.75" in captured.out


def test_cli_quick_no_crash(tmp_path, monkeypatch):
    """Smoke-test: --quick with empty JSON dir exits without error."""
    import subprocess, sys
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    test_ann = tmp_path / "test_annotations.json"
    test_ann.write_text("{}")
    ref_ann = tmp_path / "reference_annotations.json"
    ref_ann.write_text("{}")
    result = subprocess.run(
        [sys.executable, "scripts/evaluate.py",
         "--quick",
         "--json-dir", str(json_dir),
         "--test-ann", str(test_ann),
         "--ref-ann", str(ref_ann)],
        capture_output=True, text=True, cwd="/mnt/e/dev/ml/assignment"
    )
    assert result.returncode == 0, result.stderr
