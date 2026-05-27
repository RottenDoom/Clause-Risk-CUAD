#!/usr/bin/env python3
"""
scripts/evaluate.py

Evaluation CLI for the contract review pipeline.

Usage:
  python scripts/evaluate.py --quick            # Stages 1-4a, no LLM (default)
  python scripts/evaluate.py --full             # All stages including LLM-as-judge
  python scripts/evaluate.py --stage retrieval  # Single stage

Requires: output/json/*.json produced by scripts/run_review.py --all-test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    OUTPUT_JSON_DIR,
    REFERENCE_ANNOTATIONS_PATH,
    TEST_ANNOTATIONS_PATH,
)
from scripts.eval.loader import load_records
from scripts.eval import clause_discovery, retrieval, interpretation, risk_rating, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the contract review pipeline.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", default=True,
                      help="Run Stages 1-4a without LLM calls (default).")
    mode.add_argument("--full", action="store_true",
                      help="Run all stages including LLM-as-judge.")
    parser.add_argument("--stage", choices=["discovery", "retrieval", "interpretation", "risk"],
                        help="Run a single stage only.")
    parser.add_argument("--json-dir", type=Path, default=OUTPUT_JSON_DIR,
                        help="Directory containing pipeline output JSON files.")
    parser.add_argument("--test-ann", type=Path, default=TEST_ANNOTATIONS_PATH,
                        help="Path to test_annotations.json.")
    parser.add_argument("--ref-ann", type=Path, default=REFERENCE_ANNOTATIONS_PATH,
                        help="Path to reference_annotations.json.")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "output" / "eval" / "results.json",
                        help="Where to write results JSON.")
    args = parser.parse_args()

    if not args.json_dir.exists():
        print(f"ERROR: --json-dir {args.json_dir} does not exist.")
        print("Run: python scripts/run_review.py --all-test")
        sys.exit(1)

    records = load_records(args.json_dir, args.test_ann, args.ref_ann)
    if not records:
        print("No records found. Check that --json-dir contains output JSON files.")
        sys.exit(0)

    print(f"Loaded {len(records)} eval records from {len({r.contract_id for r in records})} contracts.")

    import json as _json
    ref_ann_data = _json.loads(args.ref_ann.read_text()) if args.ref_ann.exists() else {}

    results: dict = {}

    run_all = args.stage is None
    run_discovery = run_all or args.stage == "discovery"
    run_retrieval = run_all or args.stage == "retrieval"
    run_interp = run_all or args.stage == "interpretation"
    run_risk = run_all or args.stage == "risk"

    if run_discovery:
        print("Stage 1: Clause discovery…")
        results["clause_discovery"] = clause_discovery.compute_metrics(records)

    if run_retrieval:
        print("Stage 2: Precedent retrieval…")
        results["retrieval"] = retrieval.compute_metrics(records, ref_ann_data)

    if run_interp:
        print("Stage 3: Interpretation completeness…")
        results["interpretation"] = interpretation.compute_metrics(records)

    if run_risk:
        if args.full:
            print("Stage 4: Risk rating (heuristic + LLM judge)…")
            from config import MODEL
            from services.generation.claude_client import ClaudeClient
            llm = ClaudeClient(model=MODEL)
            results["risk_rating"] = risk_rating.run_llm_judge(records, llm=llm)
        else:
            print("Stage 4: Risk rating (heuristic only)…")
            results["risk_rating"] = risk_rating.RiskMetrics(
                heuristic=risk_rating.compute_heuristic_metrics(records)
            )

    report.print_table(results)
    report.write_results_json(results, args.out)
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
