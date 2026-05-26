# Contract Clause Risk Review Agent

An agentic pipeline that analyses commercial contracts against four high-risk clause families, retrieves precedents from 80 reference contracts [CUAD Dataset](https://github.com/TheAtticusProject/cuad), and generates structured risk cards via LLM calls.

---

## Features
Following are some of the features I have taken into account with respect to the constraints.

- **Clause Discovery** — finds clause spans using local sentence embeddings. No LLM, no external DB. Retries with a broader anchor set before giving up.
- **Precedent Retrieval** — queries a Pinecone (or ChromaDB) vector index of 80 reference contracts. Returns up to 3 similar and 3 contrasting precedents per clause family.
- **Structured Interpretation** — regex heuristics for deterministic fields + one targeted LLM call for semantic fields (consent scope, acquirer binding, exclusivity scope).
- **LLM Risk Rating** — one LLM call per clause family produces `risk_rating`, `risk_rationale`, and `confidence_uncertainty_notes`. Retries once on JSON parse failure; degrades gracefully on second failure.
- **Rule-Based Aggregate Risk** — `max(cards)` with no extra LLM call. On-demand summary available separately.
- **Dual Vector Backend** — Pinecone (cloud, free tier) and ChromaDB (local) are both supported. `build_index.py` can populate both in parallel.
- **FastAPI Backend** — async job queue (ThreadPoolExecutor), in-process job store, polling endpoint.

---

## Pipeline
The architecture for the project follows a Agentic Loop hueristic. The pipeline tests
clause discovery using `anchor queries`, if the score is below a threshold the agent retries using a broader anchor length. The next step is precedent retrieval using the found clause and search for similar and contrasting clauses.  Any failure is given retry logic and a failure handler as well. Final llm call is done on the retrieved information for final interpretation and risk rating. The summary since not really required is not explicitely done in the loop but can be done towards the end as well. Jinja2 Templates are used for now.

```
Raw contract text (.txt)
        │
        ▼  repeated for each selected clause family
┌───────────────────────────────────────────────────────┐
│ Step 1 — Clause Discovery                             │
│   Local embeddings only (sentence-transformers)       │
│   5 anchor queries → cosine similarity → top-K merge  │
│   Retry: broad anchors + relaxed threshold (0.85×)    │
│   LLM calls: 0                                        │
└───────────────────────────────────────────────────────┘
        │ {found, extracted_text, score}
        ▼
┌───────────────────────────────────────────────────────┐
│ Step 2 — Precedent Retrieval                          │
│   Pinecone / ChromaDB  (skipped if clause not found)  │
│   Similar:     1 batch LLM call → why_similar ×N     │
│   Contrasting: 1 batch LLM call → top-3 selection    │
│   LLM calls: 2                                        │
└───────────────────────────────────────────────────────┘
        │ similar[≤3], contrasting[≤3]
        ▼
┌───────────────────────────────────────────────────────┐
│ Step 3 — Interpretation + Risk Rating                 │
│   Pass 1: regex/keyword heuristics (no LLM)          │
│   Pass 2: 0–1 LLM call for semantic fields            │
│   Risk Rating: 1 LLM call, 1 retry on parse failure  │
│   LLM calls: 1–3                                      │
└───────────────────────────────────────────────────────┘
        │ ClauseCard
        ▼  after all families
┌───────────────────────────────────────────────────────┐
│ Step 4 — Aggregation                                  │
│   Rule-based overall risk (max of cards)              │
│   LLM calls: 0                                        │
│                                                       │
│   Optional: POST /review/{id}/summarize               │
│   1 LLM call → overall_summary + top_red_flags        │
└───────────────────────────────────────────────────────┘
        │ ContractReviewOutput
        ▼
  output/json/{contract_id}.json
  output/html/{contract_id}.html
```

**LLM call budget per contract (all 4 families found):**

| Step | Calls (happy path) | Calls (worst case) |
|------|-------------------|--------------------|
| Precedent retrieval × 4 | 8 | 8 |
| Interpretation semantic × 3 | 3 | 3 |
| Risk rating × 4 | 4 | 8 (all retry) |
| Aggregation summary (optional) | 1 | 1 |
| **Total** | **15** | **19** |

---

## Repository Layout

```
├── config.py                        # All paths, constants, model names
│
├── agent/
│   ├── models.py                    # Pydantic schemas
│   ├── loop.py                      # Orchestrator — Steps 1–4 with retry
│   ├── clause_discovery.py          # Step 1: embedding-based clause finder
│   ├── interpretation.py            # Step 3a: heuristic + LLM interpretation
│   ├── precedent_retrieval.py       # Step 2: Pinecone/ChromaDB queries
│   ├── risk_rating.py               # Step 3b: LLM risk card + retry
│   └── summarizer.py                # Step 4: rule-based + optional LLM summary
│
├── services/
│   ├── generation/
│   │   ├── base.py                  # Abstract LLMClient
│   │   └── claude_client.py         # Anthropic Claude implementation
│   ├── retrieval/
│   │   ├── retriever.py             # Pinecone query interface (active)
│   │   └── retriever_chroma.py      # ChromaDB query interface (backup)
│   ├── indexing/
│   │   ├── indexer.py               # Pinecone indexing service
│   │   └── indexer_chroma.py        # ChromaDB indexing service (backup)
│   ├── ingestion/
│   │   └── ingestor.py              # .txt → normalised plain text
│   ├── output/
│   │   ├── json_writer.py           # ContractReviewOutput → JSON
│   │   └── html_renderer.py         # Jinja2 HTML report
│   └── logging_setup.py             # Loguru config + stdlib intercept
│
├── api/
│   ├── routes.py                    # FastAPI app + all endpoints
│   └── models.py                    # Job/response Pydantic models
│
├── scripts/
│   ├── prepare_data.py              # 80/20 CUAD split, annotation JSONs
│   ├── build_index.py               # Embed + upsert → Pinecone and/or ChromaDB
│   ├── run_review.py                # CLI entrypoint
│   ├── test_api.sh                  # End-to-end curl test script
│   └── deploy/
│       ├── setup_ec2.sh             # One-time EC2 bootstrap
│       ├── deploy.sh                # Pull + restart
│       ├── nginx.conf               # Nginx reverse proxy config
│       └── contract-review.service  # Systemd unit file
│
├── templates/report.html            # Jinja2 HTML report template
├── data/
│   ├── reference/                   # 80 reference .txt files
│   ├── test/                        # 20 test .txt files
│   └── chroma_db/                   # ChromaDB persistence (gitignored)
└── output/
    ├── json/                        # Review output JSON files
    ├── html/                        # Review output HTML reports
    └── logs/                        # Rotating loguru logs
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Anthropic API key
- Pinecone API key (free tier — [pinecone.io](https://www.pinecone.io))
- CUAD dataset (`full_contracts_txt/` and `master_clauses.csv` in `data/cuad_raw/`)

### 1. Install

```bash
git clone <repo-url>
cd contract-review

uv venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

uv pip install -r requirements.txt
```

### 2. Environment Variables

Create `.env` in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...
PINECONE_API_KEY=pc-...
PINECONE_INDEX_NAME=cuad-contracts
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Optional
LOG_LEVEL=INFO                    # DEBUG for token counts and chunk scores
```

### 3. Prepare Data (one time)

```bash
# Split CUAD into 80 reference + 20 test contracts
python3 scripts/prepare_data.py
```

### 4. Build the Vector Index (one time)

```bash
# Both backends in parallel (fastest)
python3 scripts/build_index.py --chromadb &
python3 scripts/build_index.py --pinecone &
wait

# Or one at a time
python3 scripts/build_index.py --pinecone     # Pinecone only
python3 scripts/build_index.py --chromadb     # ChromaDB only
python3 scripts/build_index.py                # both sequentially
```

First Pinecone run provisions the `cuad-contracts` index (~60 s). Subsequent runs upsert directly (idempotent).

### 5. Run a Review (CLI)

```bash
python3 scripts/run_review.py --contract data/test/SomeContract.txt

# All 20 test contracts
python3 scripts/run_review.py --all-test
```

### 6. Start the API Server

```bash
uvicorn api.routes:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the auto-generated Swagger UI.

---

## API Reference

All endpoints are served from `http://localhost:8000`.

### `POST /review`

Submit a contract for review. Returns a `job_id` immediately; the review runs asynchronously.

| Field | Type | Description |
|-------|------|-------------|
| `contract_text` | Form string | Raw contract text (mutually exclusive with `file`) |
| `file` | File upload | `.txt` file (mutually exclusive with `contract_text`) |
| `families` | Form string | Comma-separated families or `"all"` (default). Valid: `assignment`, `change_of_control`, `termination`, `exclusivity` |
| `model` | Form string | Claude model ID (default: `claude-haiku-4-5-20251001`). See `GET /models`. |

```bash
# Text input, single family
curl -X POST http://localhost:8000/review \
  -F "contract_text=..." \
  -F "families=termination" \
  -F "model=claude-sonnet-4-6"

# File upload, all families
curl -X POST http://localhost:8000/review \
  -F "file=@data/test/MyContract.txt"
```

**Response `202`:**
```json
{
  "job_id": "abc-123",
  "contract_id": "MyContract_7f0f4141",
  "families": ["termination"],
  "model": "claude-sonnet-4-6",
  "status": "pending"
}
```

---

### `GET /review/{job_id}`

Poll job status. When `status == "done"` the full `result` object is included.

**Status values:** `pending` → `running` → `done` | `failed`

```bash
curl http://localhost:8000/review/abc-123
```

---

### `POST /review/{job_id}/summarize`

Generate `overall_summary` and `top_red_flags` on demand (1 LLM call). Updates the stored job result. Only available when `status == "done"`.

```bash
curl -X POST http://localhost:8000/review/abc-123/summarize
```

---

### `GET /review/{job_id}/report`

Returns the rendered HTML report. Only available when `status == "done"`.

---

### `GET /models`

Lists available Claude models and the current default.

### `GET /families`

Lists the four clause families with display names and descriptions.

### `GET /health`

Liveness check. Returns `{"status": "ok"}`.

---

## Testing the API

```bash
# Local server (default)
./scripts/test_api.sh

# Against a deployed server
BASE_URL=http://YOUR_EC2_IP ./scripts/test_api.sh

# Single family, cheapest model
FAMILIES=termination MODEL=claude-haiku-4-5-20251001 ./scripts/test_api.sh

# From a real contract file
CONTRACT_FILE=data/test/SomeContract.txt ./scripts/test_api.sh
```

---

## Evaluation

After reviewing all 20 test contracts:

```bash
python3 scripts/evaluate.py
```

Targets:

| Metric | Target |
|--------|--------|
| Clause Found Accuracy | > 0.70 |
| Clause Discovery Recall | > 0.65 |
| Clause Discovery Precision | > 0.60 |
| Risk Distribution | Not all same level |

---


## Configuration

All tunable constants live in `config.py`. The most commonly adjusted:

| Constant | Default | Effect |
|----------|---------|--------|
| `DISCOVERY_MIN_SCORE` | `0.30` | Lower → higher clause recall, lower precision |
| `PRECEDENT_SIMILAR_TOP_K` | `3` | Similar precedents returned per family |
| `PRECEDENT_CONTRAST_FETCH_K` | `10` | Contrast candidates over-fetched before LLM selects 3 |
| `MODEL` | `claude-sonnet-4-6` | Default Claude model for the pipeline |
| `MAX_TOKENS` | `1024` | Token ceiling for all LLM calls |
| `PINECONE_INDEX_NAME` | `cuad-contracts` | Overridable via env var |

---

## Known Limitations

| Issue | Workaround |
|-------|-----------|
| PDF ingestion not supported | Convert to `.txt` before submitting |
| Job state lost on server restart | Restart clears all in-flight and completed jobs |
| Families processed sequentially | Run single-family reviews if latency is critical |
| Embedding anchors re-computed each run | Anchor embeddings are not cached between requests |
