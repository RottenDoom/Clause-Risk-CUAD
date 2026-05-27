"""scripts/eval/loader.py

Reads pipeline output JSONs and annotation files.
This is the ONLY module permitted to read test_annotations.json (C8).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EvalRecord:
    contract_id: str
    family: str
    # Ground truth
    gt_clause_present: bool
    gt_clause_text: Optional[str]
    # Pipeline output
    clause_found: bool
    extracted_clause_text: Optional[str]
    structured_interpretation: Optional[dict]
    similar_contract_ids: list[str]          # from similar_precedents[*].contract_id
    llm_generated_risk_rating: Optional[str]
    risk_rationale: Optional[str]
    confidence_uncertainty_notes: list[str] = field(default_factory=list)
    discovery_score: float = 0.0
    # Derived from reference_annotations
    relevant_reference_ids: frozenset = field(default_factory=frozenset)


def load_records(
    output_json_dir: Path,
    test_annotations_path: Path,
    reference_annotations_path: Path,
) -> list[EvalRecord]:
    """
    Returns one EvalRecord per (contract, family) pair present in output JSONs.
    Only families present in the pipeline output are included — partial runs are supported.
    """
    test_ann: dict = json.loads(test_annotations_path.read_text())
    ref_ann: dict = json.loads(reference_annotations_path.read_text())

    # Build relevance sets: family → set of ref contract IDs with non-null annotation
    relevance: dict[str, frozenset] = {}
    all_families = {"assignment", "change_of_control", "termination", "exclusivity"}
    for family in all_families:
        relevance[family] = frozenset(
            cid for cid, fam_map in ref_ann.items()
            if fam_map.get(family) is not None
        )

    records: list[EvalRecord] = []
    for json_file in sorted(output_json_dir.glob("*.json")):
        output = json.loads(json_file.read_text())
        contract_id = output["contract_id"]
        contract_gt = test_ann.get(contract_id, {})

        for card in output.get("clause_cards", []):
            family = card["clause_family"]
            gt_text = contract_gt.get(family)
            records.append(EvalRecord(
                contract_id=contract_id,
                family=family,
                gt_clause_present=gt_text is not None,
                gt_clause_text=gt_text,
                clause_found=card["clause_found"],
                extracted_clause_text=card.get("extracted_clause_text"),
                structured_interpretation=card.get("structured_interpretation"),
                similar_contract_ids=[
                    p["contract_id"] for p in card.get("similar_precedents", [])
                ],
                llm_generated_risk_rating=card.get("llm_generated_risk_rating"),
                risk_rationale=card.get("risk_rationale"),
                confidence_uncertainty_notes=card.get("confidence_uncertainty_notes", []),
                discovery_score=card.get("discovery_score", 0.0),
                relevant_reference_ids=relevance.get(family, frozenset()),
            ))
    return records
