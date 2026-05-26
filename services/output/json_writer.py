"""
services/output/json_writer.py

Serialises a ContractReviewOutput to a JSON file.
"""

from pathlib import Path

from agent.models import ContractReviewOutput
from config import OUTPUT_JSON_DIR


def write_json(output: ContractReviewOutput, dest_dir: Path = OUTPUT_JSON_DIR) -> Path:
    """
    Write output to dest_dir/{contract_id}.json.
    Validates that clause_cards contains exactly 4 items before writing.
    Returns the path of the written file.
    """
    n = len(output.clause_cards)
    if not 1 <= n <= 4:
        raise ValueError(
            f"ContractReviewOutput must have 1–4 clause cards, got {n}"
        )

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    out_path = dest_dir / f"{output.contract_id}.json"
    out_path.write_text(output.model_dump_json(indent=2), encoding="utf-8")
    return out_path
