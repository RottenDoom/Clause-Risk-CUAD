# Evaluation & Testing Guide

This document explains the benchmark metrics used in this project, how to interpret them, and how to thoroughly test the pipeline.

---

## Table of Contents

1. [What We Are Measuring (and Why)](#1-what-we-are-measuring-and-why)
2. [Stage 1 — Clause Discovery Metrics](#2-stage-1--clause-discovery-metrics)
3. [Stage 2 — Retrieval Metrics](#3-stage-2--retrieval-metrics)
4. [Stage 3 — Interpretation Completeness](#4-stage-3--interpretation-completeness)
5. [Stage 4 — Risk Rating Metrics](#5-stage-4--risk-rating-metrics)
6. [How to Run the Evaluation](#6-how-to-run-the-evaluation)
7. [Reading the Results Table](#7-reading-the-results-table)
8. [Testing Guide](#8-testing-guide)

---

## 1. What We Are Measuring (and Why)

The pipeline has four steps. Each step is evaluated independently so you know exactly where things break down.

```
Contract text
    │
    ▼ Step 1: Did we find the clause at all?          → Stage 1 (Discovery)
    │
    ▼ Step 2: Did we retrieve useful precedents?       → Stage 2 (Retrieval)
    │
    ▼ Step 3: Did we extract all structured fields?    → Stage 3 (Interpretation)
    │
    ▼ Step 4: Is the risk rating defensible?           → Stage 4 (Risk Rating)
```

**Ground truth source:** CUAD (Contract Understanding Atticus Dataset) annotations for 20 held-out test contracts. A non-null annotation means the clause is present in that contract.

**Key constraint (C8):** `test_annotations.json` is only read by `scripts/eval/loader.py`. No agent module ever sees it.

---

## 2. Stage 1 — Clause Discovery Metrics

**Question:** For each (contract, clause family) pair, did the pipeline correctly decide whether the clause exists?

The pipeline outputs `clause_found: True/False`. Ground truth is whether CUAD has a non-null annotation for that family.

### Confusion Matrix Terms

| | Pipeline says: Found | Pipeline says: Not Found |
|---|---|---|
| **Actually present** | TP (True Positive) | FN (False Negative) |
| **Actually absent** | FP (False Positive) | TN (True Negative) |

**Example:** Contract has an Assignment clause. Pipeline finds it → TP. Pipeline misses it → FN.

### Metrics

**Accuracy** = (TP + TN) / total

> Plain English: "What fraction of all decisions were correct?"
> Target: > 0.70

**Precision** = TP / (TP + FP)

> Plain English: "Of all the clauses the pipeline *claims* to have found, how many were real?"
> Low precision = many false alarms. A jumpy pipeline that calls everything a clause would have low precision.
> Target: > 0.60

**Recall** = TP / (TP + FN)

> Plain English: "Of all the clauses that *actually exist*, how many did the pipeline catch?"
> Low recall = lots of missed clauses. This is the more dangerous failure in legal review — missing a clause is worse than a false alarm.
> Target: > 0.65

**F1** = 2 × P × R / (P + R)

> Plain English: "Harmonic mean of precision and recall." Balances both equally.
> Use when you don't have a strong preference between P and R.

**F2** = 5 × P × R / (4P + R)

> Plain English: "Like F1 but recall counts twice as much as precision."
> Sourced from ContractEval. Justified because missing a clause (FN) has worse consequences than flagging a non-existent one (FP). Use F2 as your primary single-number summary.

**Laziness Rate** = 1 − Recall

> Plain English: "What fraction of existing clauses did the pipeline skip?"
> Named by ContractEval. A lazy pipeline has high laziness rate — it gives up too easily.

**P@80R (Precision at 80% Recall)**

> Plain English: "If we lower our detection threshold until we catch 80% of all clauses, what is our precision at that operating point?"
>
> This is the CUAD paper's headline metric. The idea: legal review requires high recall (you can't miss too many clauses), so the question is: *given that we accept the cost of catching 80% of clauses, how many false alarms do we get?*
>
> If `None` — the pipeline cannot reach 80% recall even at its most aggressive threshold. This is a serious problem.

**How P@80R is computed:** The `discovery_score` (embedding cosine similarity) is swept from high to low as a classification threshold. At each threshold, we compute recall. The first threshold where recall ≥ 0.80 is the operating point; we report the precision there.

---

## 3. Stage 2 — Retrieval Metrics

**Question:** When the pipeline retrieved similar precedents for a clause, were those precedents actually relevant?

**Relevance oracle (binary):** A reference contract is relevant for family F if `reference_annotations.json[contract_id][F]` is non-null. This requires no human labeling — it derives directly from CUAD.

**Scope:** Only evaluated for (contract, family) pairs where `clause_found = True`. If the clause wasn't found, retrieval never ran.

### Metrics

**Recall@3**

> Plain English: "Did at least one relevant precedent appear in the top 3 retrieved results?"
> Binary hit/miss per query, averaged. Sourced from ACORD benchmark.
>
> Example: Retrieved = [RefA, RefB, RefC]. Relevant set = {RefB, RefD}.
> RefB is in top 3 → Recall@3 = 1.0 for this query.
>
> If all retrieved contracts are irrelevant → Recall@3 = 0.0.

**MRR (Mean Reciprocal Rank)**

> Plain English: "On average, how high did the first relevant result rank?"
> For each query: reciprocal rank = 1 / (rank of first relevant hit). If no hit, reciprocal rank = 0.
>
> MRR = 1.0 → relevant result always at position 1.
> MRR = 0.5 → relevant result typically at position 2.
> MRR = 0.0 → never retrieved anything relevant.
>
> Sourced from ACORD and LegalBench-RAG. LegalBench-RAG threshold: MRR < 0.4 = poor quality retrieval.

**nDCG@3 (Normalized Discounted Cumulative Gain at k=3)**

> Plain English: "How well does the ranking of results match the ideal ranking?"
> A relevant result at rank 1 is worth more than one at rank 3. nDCG penalizes relevant results that appear lower in the list.
>
> DCG@3 = Σ gain(rank_i) / log2(rank_i + 1)  for i in [1..3]
> nDCG@3 = DCG@3 / IDCG@3  (divided by ideal DCG)
>
> nDCG = 1.0 → perfect ranking. nDCG = 0.0 → no relevant results in top 3.
> LegalBench-RAG threshold: nDCG > 0.7 = production-grade.
>
> Example:
> - Retrieved = [RefA (relevant), RefB (not), RefC (relevant)]
> - DCG = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
> - IDCG (ideal: both relevant at 1 and 2) = 1/log2(2) + 1/log2(3) = 1.0 + 0.631 = 1.631
> - nDCG = 1.5 / 1.631 ≈ 0.92

**Avg Semantic Similarity**

> Plain English: "How similar (by embedding cosine distance) are the retrieved clause texts to the test clause text?"
> Secondary signal — does not require ground-truth relevance labels. A score near 1.0 means the embedding retrieval is finding semantically close clauses. A low score suggests the ChromaDB index or embeddings are off.

---

## 4. Stage 3 — Interpretation Completeness

**Question:** For each clause card, did the LLM fill in all the expected structured fields?

**Expected fields per family** (defined in `scripts/eval/interpretation.py`):

| Family | Expected fields |
|--------|----------------|
| assignment | `assignment_restricted`, `consent_required`, `restricted_party`, `exceptions_detected`, `confidence` |
| change_of_control | `trigger_on_control_change`, `termination_right`, `consent_required`, `acquirer_bound`, `exceptions_detected`, `confidence` |
| termination | `termination_for_convenience`, `notice_period_days`, `cure_period_days`, `mutual_termination_right`, `grounds_detected`, `confidence` |
| exclusivity | `exclusivity_granted`, `scope`, `geographic_limitation`, `non_compete_present`, `duration_mentioned`, `confidence` |

**Field completeness rate** = (fields present as keys) / (total expected fields), averaged across all cards.

> Note: A field is "present" if its key appears in `structured_interpretation`, even if the value is `None` (which is a valid value for optional numeric fields like `notice_period_days`).

**Scope:** Only evaluated for cards where `clause_found = True` and `structured_interpretation` is not null.

**Why this metric?** There is no field-level ground truth in CUAD for our interpretation schema — CUAD provides full clause text, not structured fields. Completeness is the best available proxy for interpretation quality without additional human labeling.

---

## 5. Stage 4 — Risk Rating Metrics

### 4a — Heuristic Agreement (--quick mode)

**Question:** Does the pipeline's LLM-generated risk rating agree with a deterministic rule-based label?

**Heuristic rules** (in `scripts/eval/risk_rating.py`):

| Family | High | Medium | Low |
|--------|------|--------|-----|
| assignment | `consent_required=True` and no `exceptions_detected` | `consent_required=True` with exceptions | `consent_required=False` |
| change_of_control | `termination_right=True` and no exceptions | `trigger_on_control_change=True` | neither |
| termination | `termination_for_convenience=True` and notice ≤ 30 days (or no notice) | `termination_for_convenience=True` with longer notice | neither |
| exclusivity | `exclusivity_granted=True` and `non_compete_present=True` | `exclusivity_granted=True` without non-compete | neither |

**Heuristic agreement %** = (records where heuristic label == pipeline label) / total.

> Not a ground truth comparison — it measures consistency between deterministic rules and the LLM's judgment. High disagreement (< 50%) means either the LLM is unreliable or the heuristics are too coarse.

### 4b — LLM-as-Judge (--full mode only)

**Question:** Is the pipeline's risk rationale accurate and its rating plausible?

For each card with `clause_found=True` and a non-null rationale, Claude is asked to:
1. Score the rationale quality 1–5
2. Judge whether the risk rating is plausible (pass/fail)
3. Produce a reference rationale

**Judge score (1–5):**
- 5 = accurate, specific, cites clause language
- 4 = mostly accurate with minor gaps
- 3 = partially correct or vague
- 2 = largely incorrect or generic
- 1 = completely wrong or empty

**Judge pass %:** Fraction of cards where Claude found the rating plausible.

**Jaccard similarity:** Word-overlap between the pipeline rationale and the judge's reference rationale.

> Jaccard(A, B) = |words_A ∩ words_B| / |words_A ∪ words_B|
> A crude but fast proxy for rationale quality. Low Jaccard means the pipeline's rationale shares little vocabulary with what the judge would say.

---

## 6. How to Run the Evaluation

### Prerequisites

All 20 test contracts must be reviewed first:

```bash
# Review all 20 test contracts (takes ~5-10 min with API calls)
python scripts/run_review.py --all-test

# Verify output exists
ls output/json/*.json | wc -l   # should print 20
```

### Running the evaluator

```bash
# Quick mode — no LLM calls, < 5 seconds
python scripts/evaluate.py --quick

# Full mode — adds LLM-as-judge (~1 API call per found clause card)
python scripts/evaluate.py --full

# Single stage (useful for debugging)
python scripts/evaluate.py --stage discovery
python scripts/evaluate.py --stage retrieval
python scripts/evaluate.py --stage interpretation
python scripts/evaluate.py --stage risk

# Custom paths (useful for testing with a subset)
python scripts/evaluate.py --quick \
  --json-dir path/to/output/json \
  --test-ann path/to/test_annotations.json \
  --ref-ann path/to/reference_annotations.json \
  --out path/to/results.json
```

Results are written to `output/eval/results.json`.

### Notebook analysis

After running `--quick` or `--full`:

```bash
jupyter notebook notebooks/eval_analysis.ipynb
```

Produces:
- Bar chart of Stage 1 metrics vs targets (green = pass, red = fail)
- Horizontal bar chart of retrieval metrics with LegalBench-RAG thresholds
- Risk distribution histogram across all 20 contracts

---

## 7. Reading the Results Table

Example output from `--quick`:

```
==========================================================
  CONTRACT REVIEW PIPELINE — EVALUATION RESULTS
==========================================================
Metric                        Value    Target    Pass
----------------------------------------------------------
  Accuracy                    0.760      0.70    ✓
  Precision                   0.720      0.60    ✓
  Recall                      0.650      0.65    ✓
  F1                          0.684       N/A    —
  F2                          0.666       N/A    —
  Laziness rate               0.350       N/A    —
  P@80% Recall                  N/A       N/A    —
  Recall@3                    0.571       N/A    —
  MRR                         0.510      0.40    ✓
  nDCG@3                      0.548      0.70    ✗
  Avg semantic sim            0.742       N/A    —
  Field completeness          0.883       N/A    —
  Heuristic agree %           0.643       N/A    —
==========================================================
```

**Reading the table:**
- `✓` = metric meets or exceeds target
- `✗` = metric is below target — this stage needs attention
- `—` = no target defined, informational only
- `N/A` in Value = metric could not be computed (e.g., P@80R when max recall < 0.80)

**If nDCG@3 < 0.70 (✗):** The retrieval ranking is below production grade. Possible causes: ChromaDB index is stale (`python scripts/build_index.py`), the embedding model isn't the right one for legal text, or `k` in the retriever is too small.

**If Recall < 0.65 (✗):** The pipeline is missing too many clauses. Check `DISCOVERY_MIN_SCORE` in `config.py` — the threshold may be too high. Try lowering it or tuning the anchor queries in `agent/clause_discovery.py`.

**If P@80R = N/A:** The pipeline can't reach 80% recall at any embedding threshold. The embedding model or anchor queries may not be suitable for the contract style.

---

## 8. Testing Guide

### Test structure

```
tests/
├── test_models.py            # Pydantic schema validation (fast, no I/O)
├── test_interpretation.py    # Interpretation field extraction
├── test_risk_rating.py       # Risk rating generation and retry logic
└── eval/
    ├── test_loader.py        # EvalRecord loading from fixtures
    ├── test_clause_discovery.py   # Stage 1 metric calculations
    ├── test_retrieval.py          # Stage 2 metric calculations
    ├── test_interpretation.py     # Stage 3 completeness calculation
    ├── test_risk_rating.py        # Stage 4 heuristics + judge
    └── test_report.py             # Report formatting + CLI smoke-test
```

### Running tests

```bash
# All tests
python -m pytest tests/ -v

# Just eval metrics (fast, no embeddings loaded)
python -m pytest tests/eval/ -v

# Just models (fastest)
python -m pytest tests/test_models.py -v

# Single test by name
python -m pytest tests/eval/test_clause_discovery.py::test_precision_at_80_recall_achievable -v

# Show full failure output
python -m pytest tests/ --tb=short

# Stop at first failure
python -m pytest tests/ -x
```

### What each test suite covers

**`tests/eval/test_loader.py`**
- Verifies that `load_records` correctly reads pipeline output JSONs and annotation files
- Checks ground truth fields are correctly set from `test_annotations.json`
- Verifies binary relevance sets are derived from `reference_annotations.json`

**`tests/eval/test_clause_discovery.py`**
- Tests accuracy, precision, recall, F1, F2, laziness rate with a known 6-record fixture (TP=3, FP=1, FN=1, TN=1)
- Tests P@80R returns `None` when recall cannot reach 80% (FN score=0.0, unrecoverable)
- Tests P@80R returns a precision value when a threshold achieves exactly 80% recall
- Tests graceful handling of empty record list

**`tests/eval/test_retrieval.py`**
- Tests Recall@3 hit and miss cases
- Tests MRR at rank 1, rank 2, and no hit
- Tests nDCG@3 for perfect ranking and partial ranking (hand-computed expected values)
- Tests that records with `clause_found=False` are skipped entirely
- Tests empty input

**`tests/eval/test_interpretation.py`**
- Tests full completeness (all keys present) = 1.0
- Tests partial completeness (one key absent) = 4/5
- Tests that `clause_found=False` records are skipped
- Tests per-family breakdown

**`tests/eval/test_risk_rating.py`**
- Tests all heuristic rules for assignment, termination (high/medium/low branches)
- Tests heuristic agreement rate with a 2-record fixture (50% agreement expected)
- Tests `clause_found=False` records are skipped in heuristic evaluation
- Tests LLM-as-judge with a `MockLLM` that returns a fixed score=4, plausible=True
- Tests Jaccard similarity: identical text = 1.0, disjoint = 0.0, partial overlap = 1/3
- Tests that records with no rationale are skipped by the judge

**`tests/eval/test_report.py`**
- Tests JSON writer produces a readable file with correct values
- Tests stdout table contains expected metric names and values
- CLI smoke-test: runs `scripts/evaluate.py --quick` with an empty JSON directory and asserts exit code 0

### How to write a new eval test

The pattern is always the same:

```python
from scripts.eval.loader import EvalRecord

def _make_record(**overrides) -> EvalRecord:
    """Create a minimal valid EvalRecord, override fields as needed."""
    defaults = dict(
        contract_id="C1", family="assignment",
        gt_clause_present=True, gt_clause_text="text",
        clause_found=True, extracted_clause_text="text",
        structured_interpretation=None, similar_contract_ids=[],
        llm_generated_risk_rating=None, risk_rationale=None,
        relevant_reference_ids=frozenset(),
    )
    return EvalRecord(**{**defaults, **overrides})

def test_my_metric():
    from scripts.eval.clause_discovery import compute_metrics
    records = [_make_record(gt_clause_present=True, clause_found=True)]
    m = compute_metrics(records)
    assert m.precision == pytest.approx(1.0)
```

### Debugging a metric that looks wrong

1. **Isolate the stage:** Run `--stage <name>` to compute only that metric.
2. **Inspect `results.json`:** The raw values are in `output/eval/results.json`.
3. **Check the loader:** Add a `print(records)` in `loader.py` temporarily to see what `EvalRecord` objects are being built.
4. **Verify output JSONs:** Check that `output/json/SomeContract.json` has the `clause_found` and `discovery_score` fields filled in correctly.
5. **Re-index if retrieval looks off:** `python scripts/build_index.py` rebuilds the ChromaDB index from scratch.

### Gotchas

- **P@80R = None is not a bug** — it means the pipeline's `discovery_score` distribution doesn't allow reaching 80% recall. The FN records have score=0.0 (pipeline never scored them highly).
- **Heuristic agreement < 100% is expected** — the LLM sees full clause text and context; the heuristic only sees structured fields. Disagreement is informative, not necessarily wrong.
- **Semantic similarity requires the embedding model to be loaded** — the first call to `embed()` in retrieval evaluation will load the sentence-transformers model (~2s). Subsequent calls use the cached singleton.
- **LLM-as-judge costs money** — `--full` makes ~1 API call per found clause card. With 20 contracts × 4 families ≈ 80 potential calls. Budget accordingly.
