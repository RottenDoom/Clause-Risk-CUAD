"""
scripts/run_review.py

CLI entrypoint that wires all agent modules together for a single contract
review or a full batch over the 20 test contracts.

Usage:
    python3 scripts/run_review.py --contract data/test/SomeContract.txt
    python3 scripts/run_review.py --all-test
    python3 scripts/run_review.py --contract path/to/contract.txt --output-dir /tmp/out
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load ANTHROPIC_API_KEY from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; fall back to env var already being set

from config import CLAUSE_FAMILIES, TEST_DIR

# Agent pipeline imports — these modules are implemented in issues 3-10.
# run_review.py is intentionally written first so it serves as the
# authoritative spec for the interface each module must expose.
from agent.models import ClauseCard, ContractReviewOutput
from agent.clause_discovery import discover_clause
from agent.interpretation import interpret_clause, get_null_interpretation
from agent.precedent_retrieval import retrieve_precedents
from agent.risk_rating import generate_risk_rating
from agent.summarizer import summarize_contract
from agent.output_writer import write_output
from services.generation.claude_client import ClaudeClient
from services.retrieval.retriever import Retriever


# ---------------------------------------------------------------------------
# Core review logic
# ---------------------------------------------------------------------------

def review_contract(contract_path: Path) -> ContractReviewOutput:
    """
    Run the full pipeline for a single contract .txt file.
    Reads only the raw text — no annotation data is accessed. 
    """
    contract_text = contract_path.read_text(encoding="utf-8", errors="ignore")
    contract_id = contract_path.stem

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: API key not set. Add it to .env or export it.")

    llm = ClaudeClient()
    retriever = Retriever()

    clause_cards: list[ClauseCard] = []

    for family in CLAUSE_FAMILIES:
        print(f"  [{family}] discovering clause...", end=" ", flush=True)
        clause_found, extracted_text, score = discover_clause(contract_text, family)
        print(f"{'FOUND' if clause_found else 'NOT FOUND'} (score={score:.3f})")

        # Interpretation and retrieval only run when the clause was found.
        # When not found, null-safe defaults are used so the card is still produced.
        if clause_found:
            interpretation = interpret_clause(extracted_text, family, llm)
        else:
            interpretation = get_null_interpretation(family)

        if clause_found and extracted_text:
            similar, contrasting = retrieve_precedents(
                extracted_text, family, retriever, llm
            )
        else:
            similar, contrasting = [], []

        print(f"  [{family}] generating risk rating...", end=" ", flush=True)
        risk_rating, rationale, notes = generate_risk_rating(
            family, extracted_text, interpretation,
            similar, contrasting, llm
        )
        print(f"{'null' if risk_rating is None else risk_rating.value}")

        clause_cards.append(ClauseCard(
            clause_family=family,
            clause_found=clause_found,
            extracted_clause_text=extracted_text,
            structured_interpretation=interpretation,
            similar_precedents=similar,
            contrasting_precedents=contrasting,
            llm_generated_risk_rating=risk_rating,
            risk_rationale=rationale,
            confidence_uncertainty_notes=notes,
        ))

    print("  Summarising contract...", end=" ", flush=True)
    overall_summary, overall_risk, red_flags = summarize_contract(
        contract_id, clause_cards, llm
    )
    print(f"overall={overall_risk.value}")

    output = ContractReviewOutput(
        contract_id=contract_id,
        overall_summary=overall_summary,
        overall_risk_rating=overall_risk,
        top_red_flags=red_flags,
        clause_cards=clause_cards,
    )

    write_output(output)
    return output


# ---------------------------------------------------------------------------
# Batch mode (--all-test)
# ---------------------------------------------------------------------------

def batch_review() -> None:
    test_files = sorted(TEST_DIR.glob("*.txt"))
    if not test_files:
        sys.exit(f"ERROR: No .txt files found in {TEST_DIR}. Run prepare_data.py first.")

    print(f"Batch reviewing {len(test_files)} test contracts...\n")
    failures: list[str] = []

    for i, contract_path in enumerate(test_files, 1):
        print(f"[{i:02d}/{len(test_files)}] {contract_path.name}")
        t0 = time.time()
        try:
            review_contract(contract_path)
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")
            failures.append(contract_path.name)

    print(f"Batch complete. {len(test_files) - len(failures)}/{len(test_files)} succeeded.")
    if failures:
        print("Failed contracts:")
        for name in failures:
            print(f"  {name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Contract Clause Risk Review Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 scripts/run_review.py --contract data/test/SomeContract.txt\n"
            "  python3 scripts/run_review.py --all-test"
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--contract", type=Path, metavar="PATH",
        help="Path to a single contract .txt file to review",
    )
    mode.add_argument(
        "--all-test", action="store_true",
        help="Review all 20 test contracts in data/test/",
    )
    args = parser.parse_args()

    if args.all_test:
        batch_review()
    else:
        contract_path: Path = args.contract
        if not contract_path.exists():
            sys.exit(f"ERROR: File not found: {contract_path}")
        if contract_path.suffix.lower() not in (".txt", ".pdf"):
            print(f"WARNING: Unexpected file extension {contract_path.suffix!r}. Proceeding anyway.")
        print(f"Reviewing: {contract_path.name}\n")
        t0 = time.time()
        output = review_contract(contract_path)
        elapsed = time.time() - t0
        print(f"\nDone in {elapsed:.1f}s")
        print(f"Overall risk: {output.overall_risk_rating.value.upper()}")


if __name__ == "__main__":
    main()
