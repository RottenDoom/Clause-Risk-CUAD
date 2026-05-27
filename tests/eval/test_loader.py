import json
import pytest
from pathlib import Path


SAMPLE_OUTPUT = {
    "contract_id": "TestContract",
    "overall_risk_rating": "high",
    "overall_summary": "",
    "top_red_flags": [],
    "clause_cards": [
        {
            "clause_family": "assignment",
            "clause_found": True,
            "extracted_clause_text": "Neither party may assign without consent.",
            "structured_interpretation": {"consent_required": True},
            "similar_precedents": [{"contract_id": "Ref1", "why_similar": "Similar restriction."}],
            "contrasting_precedents": [],
            "llm_generated_risk_rating": "high",
            "risk_rationale": "Strict consent required.",
            "confidence_uncertainty_notes": [],
            "discovery_score": 0.45,
        },
        {
            "clause_family": "termination",
            "clause_found": False,
            "extracted_clause_text": None,
            "structured_interpretation": None,
            "similar_precedents": [],
            "contrasting_precedents": [],
            "llm_generated_risk_rating": None,
            "risk_rationale": None,
            "confidence_uncertainty_notes": [],
            "discovery_score": 0.0,
        },
    ],
}

SAMPLE_TEST_ANNOTATIONS = {
    "TestContract": {
        "assignment": "Neither party may assign...",
        "change_of_control": None,
        "termination": None,
        "exclusivity": None,
    }
}

SAMPLE_REF_ANNOTATIONS = {
    "Ref1": {
        "assignment": "Assignment is prohibited without prior written consent.",
        "change_of_control": None,
        "termination": None,
        "exclusivity": None,
    },
    "Ref2": {
        "assignment": None,
        "change_of_control": "Change of control triggers termination.",
        "termination": None,
        "exclusivity": None,
    },
}


@pytest.fixture
def tmp_dirs(tmp_path):
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    (json_dir / "TestContract.json").write_text(json.dumps(SAMPLE_OUTPUT))

    test_ann = tmp_path / "test_annotations.json"
    test_ann.write_text(json.dumps(SAMPLE_TEST_ANNOTATIONS))

    ref_ann = tmp_path / "reference_annotations.json"
    ref_ann.write_text(json.dumps(SAMPLE_REF_ANNOTATIONS))

    return json_dir, test_ann, ref_ann


def test_load_records_count(tmp_dirs):
    from scripts.eval.loader import load_records
    json_dir, test_ann, ref_ann = tmp_dirs
    records = load_records(json_dir, test_ann, ref_ann)
    # 2 clause cards in the output JSON → 2 records
    assert len(records) == 2


def test_load_records_gt_present(tmp_dirs):
    from scripts.eval.loader import load_records
    json_dir, test_ann, ref_ann = tmp_dirs
    records = load_records(json_dir, test_ann, ref_ann)
    assignment_rec = next(r for r in records if r.family == "assignment")
    assert assignment_rec.gt_clause_present is True
    assert assignment_rec.gt_clause_text == "Neither party may assign..."


def test_load_records_gt_absent(tmp_dirs):
    from scripts.eval.loader import load_records
    json_dir, test_ann, ref_ann = tmp_dirs
    records = load_records(json_dir, test_ann, ref_ann)
    term_rec = next(r for r in records if r.family == "termination")
    assert term_rec.gt_clause_present is False
    assert term_rec.gt_clause_text is None


def test_load_records_pipeline_fields(tmp_dirs):
    from scripts.eval.loader import load_records
    json_dir, test_ann, ref_ann = tmp_dirs
    records = load_records(json_dir, test_ann, ref_ann)
    assignment_rec = next(r for r in records if r.family == "assignment")
    assert assignment_rec.clause_found is True
    assert assignment_rec.discovery_score == pytest.approx(0.45)
    assert assignment_rec.llm_generated_risk_rating == "high"
    assert assignment_rec.similar_contract_ids == ["Ref1"]


def test_load_records_relevant_reference_ids(tmp_dirs):
    from scripts.eval.loader import load_records
    json_dir, test_ann, ref_ann = tmp_dirs
    records = load_records(json_dir, test_ann, ref_ann)
    assignment_rec = next(r for r in records if r.family == "assignment")
    # Only Ref1 has a non-null assignment annotation
    assert assignment_rec.relevant_reference_ids == frozenset({"Ref1"})
