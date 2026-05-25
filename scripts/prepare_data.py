"""
scripts/prepare_data.py

Entry point for the offline data preparation stage. Run this once before
build_index.py. It:

  1. Downloads CUAD from HuggingFace into the datasets cache (first run ~400MB).
  2. Extracts contract texts and clause annotations for our four target families.
  3. Deterministically selects 100 contracts (seed=42), splits 80 reference / 20 test.
  4. Writes contract .txt files to data/reference/ and data/test/.
  5. Writes data/reference_annotations.json (accessible to the agent pipeline).
  6. Writes data/test_annotations.json (HELD OUT — only evaluate.py reads this).
  7. Prints a per-family annotation coverage summary.

Usage:
    python3 scripts/prepare_data.py
"""

import json
import random
import re
import sys
from pathlib import Path

# Allow running as a top-level script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CLAUSE_FAMILIES,
    CUAD_COLUMN_MAP,
    FULL_CONTRACTS_DIR,
    REFERENCE_ANNOTATIONS_PATH,
    REFERENCE_COUNT,
    REFERENCE_DIR,
    SPLIT_SEED,
    TEST_ANNOTATIONS_PATH,
    TEST_DIR,
    TOTAL_CONTRACTS_TO_USE,
    CUAD_RAW_DIR
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Flat set of CUAD column names we track — built from config to stay in sync.
_TARGET_CATEGORIES: set[str] = {col for cols in CUAD_COLUMN_MAP.values() for col in cols}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_id(title: str) -> str:
    """Convert a CUAD contract title/filename to a filesystem-safe identifier."""
    return re.sub(r"[^\w.\-]", "_", title)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def ensure_cuad_downloaded() -> None:
    """
    Download CUAD TXT files and the annotation CSV from HuggingFace using
    snapshot_download, which lets us pull only the file types we need.

    We switched away from `datasets.load_dataset` because the
    theatticusproject/cuad dataset was restructured to serve PDFs, which
    requires pdfplumber to decode — an unnecessary dependency for this project.
    snapshot_download fetches the raw repository files instead.

    Skips the download entirely if TXT and CSV files are already present.
    """
    txt_files = list(CUAD_RAW_DIR.rglob("*.txt"))
    csv_files = list(CUAD_RAW_DIR.rglob("*.csv"))

    if txt_files and csv_files:
        print(f"  CUAD already present: {len(txt_files)} TXT, {len(csv_files)} CSV.")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "ERROR: huggingface_hub not found.\n"
            "  Run: uv pip install huggingface_hub"
        )

    print("Downloading CUAD TXT files and annotations from HuggingFace (skipping PDFs)...")
    CUAD_RAW_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="theatticusproject/cuad",
        repo_type="dataset",
        local_dir=str(CUAD_RAW_DIR),
        allow_patterns=["*.txt", "*.csv", "**/*.txt", "**/*.csv"],
        ignore_patterns=["*.pdf", "**/*.pdf"],
    )
    print(f"  Download complete → {CUAD_RAW_DIR}")


# ---------------------------------------------------------------------------
# Annotation CSV reader
# ---------------------------------------------------------------------------

def _read_annotations_csv(csv_path: Path) -> dict[str, dict[str, str | None]]:
    """
    Parse master_clauses.csv into {contract_id: {cuad_category: text | None}}.

    Handles two common ID column names and falls back to case-insensitive
    column matching for the clause columns — minor casing differences between
    dataset versions (e.g. "Anti-assignment" vs "Anti-Assignment") are common.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    print(f"  Annotation CSV: {csv_path.name} ({len(df)} rows, {len(df.columns)} cols)")

    # Identify the contract ID column
    possible_id_cols = ["Filename", "Document Name", "filename", "contract_name", "Title", "title"]
    id_col = next((c for c in possible_id_cols if c in df.columns), None)
    if id_col is None:
        # Print all columns so the user can identify the right one
        print(f"  All CSV columns:\n    {df.columns.tolist()}")
        sys.exit(
            "ERROR: Cannot identify the contract ID column in the annotation CSV.\n"
            "  Add the correct column name to possible_id_cols in _read_annotations_csv()."
        )

    # Case-insensitive column lookup so minor casing differences don't break us
    col_lower: dict[str, str] = {c.lower(): c for c in df.columns}

    annotations: dict[str, dict[str, str | None]] = {}
    for _, row in df.iterrows():
        # Use only the stem so we match whether the CSV stores "Contract.pdf" or "Contract"
        raw_id = str(row[id_col])
        cid = _safe_id(Path(raw_id).stem)

        raw_clauses: dict[str, str | None] = {}
        for category in _TARGET_CATEGORIES:
            actual_col = col_lower.get(category.lower())
            if actual_col:
                val = row.get(actual_col)
                raw_clauses[category] = str(val).strip() if isinstance(val, str) and val.strip() else None
            else:
                raw_clauses[category] = None

        annotations[cid] = raw_clauses

    matched = sum(1 for a in annotations.values() if any(v for v in a.values()))
    print(f"  {len(annotations)} contracts in CSV, {matched} with ≥1 target annotation.")
    return annotations


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_cuad() -> dict[str, dict]:
    """
    Ensure CUAD is downloaded, then load contract texts and annotations from disk.

    Returns {contract_id: {"text": str, "raw_clauses": {category: text | None}}}
    """
    ensure_cuad_downloaded()

    # CUAD organises contracts in subdirectories (Part_I/, Part_II/, etc.)
    # Restrict to paths whose immediate parent looks like a CUAD contract directory.
    all_txts = [
        p for p in CUAD_RAW_DIR.rglob("*.txt")
        if "full_contract" in p.parent.name.lower()
    ]
    if not all_txts:
        # Broader fallback: any TXT file anywhere under the directory
        all_txts = list(CUAD_RAW_DIR.rglob("*.txt"))

    print(f"  Found {len(all_txts)} contract TXT files.")
    if not all_txts:
        sys.exit(f"ERROR: No TXT files found under {CUAD_RAW_DIR}.")

    contract_texts: dict[str, str] = {
        _safe_id(p.stem): p.read_text(encoding="utf-8", errors="replace")
        for p in all_txts
    }

    # Prefer master_clauses.csv; fall back to any CSV at the repo root
    csv_candidates = sorted(CUAD_RAW_DIR.rglob("master_clauses.csv"))
    if not csv_candidates:
        csv_candidates = sorted(CUAD_RAW_DIR.glob("*.csv"))
    if not csv_candidates:
        sys.exit("ERROR: No annotation CSV found. Re-run without skipping the download.")

    raw_annotations = _read_annotations_csv(csv_candidates[0])

    empty_clauses: dict[str, str | None] = {cat: None for cat in _TARGET_CATEGORIES}
    contracts: dict[str, dict] = {
        cid: {
            "text": text,
            "raw_clauses": raw_annotations.get(cid, empty_clauses),
        }
        for cid, text in contract_texts.items()
    }

    print(f"  {len(contracts):,} contracts loaded.")
    return contracts


# ---------------------------------------------------------------------------
# Family annotation building
# ---------------------------------------------------------------------------

def build_family_annotations(contracts: dict) -> dict[str, dict]:
    """
    Convert raw CUAD category texts into our four-family annotation schema.

    For 'exclusivity', CUAD_COLUMN_MAP maps two categories (Exclusivity and
    Non-Compete). If either is non-empty the family is annotated; both texts
    are concatenated when both are present.

    Returns {contract_id: {family: "clause text" | None}}
    """
    annotations: dict[str, dict] = {}
    for cid, data in contracts.items():
        raw = data["raw_clauses"]
        entry: dict[str, str | None] = {}
        for family, cuad_cols in CUAD_COLUMN_MAP.items():
            texts = [raw.get(col) for col in cuad_cols if raw.get(col)]
            entry[family] = "\n".join(texts) if texts else None
        annotations[cid] = entry
    return annotations


# ---------------------------------------------------------------------------
# Contract selection and splitting
# ---------------------------------------------------------------------------

def select_contracts(
    contracts: dict, annotations: dict
) -> tuple[list[str], list[str]]:
    """
    Filter to contracts with at least one annotated clause, shuffle with a
    fixed seed, and return (reference_ids, test_ids).

    Contracts with zero annotations are excluded: they would pad the reference
    index with empty collections and skew evaluation recall metrics.

    The sort-then-shuffle pattern guarantees the same split regardless of the
    dict insertion order, which can vary across Python/HuggingFace versions.
    """
    annotated = sorted(
        cid for cid in contracts
        if any(annotations[cid][f] for f in CLAUSE_FAMILIES)
    )
    print(f"  Contracts with ≥1 annotation: {len(annotated):,}")

    if len(annotated) < TOTAL_CONTRACTS_TO_USE:
        sys.exit(
            f"ERROR: Only {len(annotated)} annotated contracts available; "
            f"need {TOTAL_CONTRACTS_TO_USE}. Check CUAD data integrity."
        )

    rng = random.Random(SPLIT_SEED)
    rng.shuffle(annotated)

    selected = annotated[:TOTAL_CONTRACTS_TO_USE]
    return selected[:REFERENCE_COUNT], selected[REFERENCE_COUNT:]


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def write_contract_files(contracts: dict, ids: list[str], dest: Path) -> None:
    """
    Write contract text files to dest/ and also persist them to
    FULL_CONTRACTS_DIR so the full corpus is available on disk for inspection.
    """
    dest.mkdir(parents=True, exist_ok=True)
    FULL_CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)

    for cid in ids:
        text = contracts[cid]["text"]

        # Persist to cuad_raw/full_contracts_txt/ for the full corpus record
        raw_path = FULL_CONTRACTS_DIR / f"{cid}.txt"
        if not raw_path.exists():
            raw_path.write_text(text, encoding="utf-8", errors="replace")

        # Write to the split directory (reference/ or test/)
        (dest / f"{cid}.txt").write_text(text, encoding="utf-8", errors="replace")


def write_annotations(annotations: dict, ids: list[str], path: Path) -> None:
    """Serialise the annotation subset for the given contract IDs to a JSON file."""
    subset = {cid: annotations[cid] for cid in ids}
    path.write_text(
        json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Coverage summary
# ---------------------------------------------------------------------------

def print_coverage_summary(
    annotations: dict, ref_ids: list[str], test_ids: list[str]
) -> None:
    """
    Print how many contracts in each split have a clause for each family.
    Low counts for a family (especially exclusivity and change_of_control)
    are expected — CUAD annotation sparsity is a known characteristic.
    Visually inspect before running build_index.py.
    """
    col_w = 24
    print("\n--- Annotation Coverage ---")
    print(f"{'Family':<{col_w}} {'Reference':>14}  {'Test':>6}")
    print("-" * (col_w + 24))
    for family in CLAUSE_FAMILIES:
        ref_n = sum(1 for cid in ref_ids if annotations[cid][family])
        test_n = sum(1 for cid in test_ids if annotations[cid][family])
        print(
            f"  {family:<{col_w - 2}}"
            f"  {ref_n:>3}/{REFERENCE_COUNT}"
            f"           {test_n:>2}/20"
        )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== CUAD Data Preparation ===\n")

    # 1. Load dataset (HuggingFace cache used on subsequent runs)
    contracts = load_cuad()

    # 2. Convert CUAD category texts to our four-family schema
    annotations = build_family_annotations(contracts)

    # 3. Deterministic filter + split
    print("\nSelecting and splitting contracts...")
    reference_ids, test_ids = select_contracts(contracts, annotations)
    print(f"  Reference: {len(reference_ids)}   Test: {len(test_ids)}")

    # 4. Write .txt files to reference/ and test/
    print("\nWriting contract files...")
    write_contract_files(contracts, reference_ids, REFERENCE_DIR)
    write_contract_files(contracts, test_ids, TEST_DIR)
    print(f"  → {REFERENCE_DIR}")
    print(f"  → {TEST_DIR}")

    # 5. Write annotation JSONs
    # reference_annotations.json is read by build_index.py and precedent_retrieval.py.
    # test_annotations.json is HELD OUT — must not be imported by any agent module.
    write_annotations(annotations, reference_ids, REFERENCE_ANNOTATIONS_PATH)
    write_annotations(annotations, test_ids, TEST_ANNOTATIONS_PATH)
    print(f"\n  reference_annotations.json → {REFERENCE_ANNOTATIONS_PATH}")
    print(f"  test_annotations.json      → {TEST_ANNOTATIONS_PATH}  [HELD OUT]")

    # 6. Coverage summary — inspect before indexing to catch sparsity issues early
    print_coverage_summary(annotations, reference_ids, test_ids)

    print("Done. Next: python3 scripts/build_index.py")


if __name__ == "__main__":
    main()
