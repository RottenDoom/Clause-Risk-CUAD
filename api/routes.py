"""
api/routes.py

FastAPI backend for the contract review frontend.

Endpoints
─────────
POST /review          Upload text or file + choose families + choose model → job_id
GET  /review/{job_id} Poll job status; returns full result when done
GET  /review/{job_id}/report  Stream rendered HTML report
GET  /models          List available Claude models
GET  /families        List available clause families
GET  /health          Liveness check

Job execution
─────────────
Jobs run in a thread pool (the review pipeline is CPU+network bound and
not async-native). Job state is held in-process; restart clears history.

Start with:
    uvicorn api.routes:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from services.logging_setup import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

from api.models import JobStatus, Job, SubmitResponse, StatusResponse
from agent.loop import ReviewLoop
from agent.models import ContractReviewOutput
from agent.summarizer import summarize_contract
from config import CLAUSE_FAMILIES, MODEL
from services.generation.claude_client import ClaudeClient
from services.output.html_renderer import render_html
from services.output.json_writer import write_json
from services.retrieval.retriever import Retriever

# ---------------------------------------------------------------------------
# App and shared singletons
# ---------------------------------------------------------------------------

app = FastAPI(title="Contract Risk Review API", version="0.2.0")


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    t0 = time.monotonic()
    response = await call_next(request)
    ms = int((time.monotonic() - t0) * 1000)
    logger.info("%s %s → %d (%dms)", request.method, request.url.path, response.status_code, ms)
    return response


_retriever: Optional[Retriever] = None
_executor = ThreadPoolExecutor(max_workers=4)

# Regex to sanity-check that submitted text looks like a contract
_CONTRACT_RE = re.compile(
    r"\b(agreement|contract|shall|party|parties|herein|whereas|term|obligations?|"
    r"liability|indemnif|terminat|assign|govern|jurisdiction)\b",
    re.IGNORECASE,
)
_MIN_CONTRACT_CHARS = 300


@app.on_event("startup")
async def _startup() -> None:
    global _retriever
    _retriever = Retriever()


# ---------------------------------------------------------------------------
# Catalogue constants — exposed via /models and /families
# ---------------------------------------------------------------------------

AVAILABLE_MODELS: list[dict[str, str]] = [
    {
        "id": "claude-haiku-4-5-20251001",
        "display_name": "Claude Haiku (default)", # for default testing
        "description": "Fastest and cheapest. Good for quick reviews.",
    },
    {
        "id": "claude-sonnet-4-6",
        "display_name": "Claude Sonnet",
        "description": "Balanced speed and quality. Recommended for most contracts.",
    }
]

_VALID_MODEL_IDS: set[str] = {m["id"] for m in AVAILABLE_MODELS}

FAMILY_CATALOGUE: list[dict[str, str]] = [
    {
        "id": "assignment",
        "display_name": "Assignment",
        "description": "Restrictions and consent requirements on transferring contract rights.",
    },
    {
        "id": "change_of_control",
        "display_name": "Change of Control",
        "description": "Rights and obligations triggered by ownership or control changes.",
    },
    {
        "id": "termination",
        "display_name": "Termination",
        "description": "Conditions, notice periods, and grounds for ending the agreement.",
    },
    {
        "id": "exclusivity",
        "display_name": "Exclusivity / Non-Compete",
        "description": "Exclusive arrangements, non-compete clauses, and territorial restrictions.",
    },
]


_jobs: dict[str, Job] = {}

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_families(raw: str | None) -> list[str]:
    """
    Parse and validate a comma-separated families string.
    'all' or empty → all 4 families.
    Returns a validated list of family ids.
    """
    if not raw or raw.strip().lower() == "all":
        return list(CLAUSE_FAMILIES)
    chosen = [f.strip() for f in raw.split(",") if f.strip()]
    invalid = [f for f in chosen if f not in CLAUSE_FAMILIES]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown clause families: {invalid}. Valid: {list(CLAUSE_FAMILIES)}",
        )
    return chosen


def _validate_model(model_id: str | None) -> str:
    if not model_id:
        return MODEL
    if model_id not in _VALID_MODEL_IDS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model: {model_id!r}. Valid: {list(_VALID_MODEL_IDS)}",
        )
    return model_id


def _validate_contract_text(text: str) -> str:
    text = text.strip()
    if len(text) < _MIN_CONTRACT_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"Contract text too short ({len(text)} chars). Minimum is {_MIN_CONTRACT_CHARS}.",
        )
    if not _CONTRACT_RE.search(text):
        raise HTTPException(
            status_code=422,
            detail=(
                "Text does not appear to be a legal contract. "
                "Make sure it contains contract-specific language "
                "(e.g. 'agreement', 'shall', 'party', 'termination')."
            ),
        )
    return text


def _contract_id_from_text(text: str) -> str:
    """Generate a stable, filesystem-safe contract_id from submitted text."""
    digest = hashlib.sha1(text[:500].encode()).hexdigest()[:8]
    slug = re.sub(r"[^\w]", "_", text[:30].strip())[:30].strip("_")
    return f"{slug}_{digest}" if slug else f"contract_{digest}"


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


def _run_review(job_id: str, contract_text: str) -> None:
    job = _jobs[job_id]
    job.status = JobStatus.running
    t0 = time.monotonic()
    logger.info(
        "job_id=%s contract_id=%s families=%s model=%s: starting",
        job_id, job.contract_id, job.families, job.model,
    )
    try:
        llm = ClaudeClient(model=job.model)
        loop = ReviewLoop(llm_client=llm, retriever=_retriever)
        output: ContractReviewOutput = loop.run(
            contract_text,
            contract_id=job.contract_id,
            families=job.families,
        )
        json_path = write_json(output)
        html_path = render_html(output)

        job.result = output.model_dump()
        job.html_path = str(html_path)
        job.status = JobStatus.done
        logger.info("job_id=%s done (%.1fs) overall_risk=%s", job_id, time.monotonic() - t0, output.overall_risk_rating)
    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.error("job_id=%s failed after %.1fs — %s", job_id, elapsed, exc, exc_info=True)
        job.status = JobStatus.failed
        job.error = str(exc)

@app.post("/review", response_model=SubmitResponse, status_code=202)
async def submit_review(
    contract_text: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = None,
    families: Optional[str] = Form(default=None),
    model: Optional[str] = Form(default=None),
) -> SubmitResponse:
    """
    Submit a contract for review.

    Supply **either** `contract_text` (raw text in the form body) **or** `file`
    (.txt upload) — not both.

    - `families`: comma-separated list of families to review, or `"all"` (default).
                  Valid values: assignment, change_of_control, termination, exclusivity.
    - `model`: Claude model ID to use (default: claude-sonnet-4-6).
               See GET /models for valid options.
    """
    # --- Input resolution ---
    if contract_text and file:
        raise HTTPException(
            status_code=422,
            detail="Supply either contract_text or file, not both.",
        )
    if not contract_text and not file:
        raise HTTPException(
            status_code=422,
            detail="Supply either contract_text or file.",
        )

    if file:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix == ".pdf":
            raise HTTPException(
                status_code=400,
                detail="PDF support is not yet available. Convert to .txt first.",
            )
        if suffix not in ("", ".txt"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type {suffix!r}. Only .txt files are accepted.",
            )
        raw = await file.read()
        contract_text = raw.decode("utf-8", errors="ignore")
        contract_id = Path(file.filename or "contract").stem
    else:
        contract_id = _contract_id_from_text(contract_text)

    # --- Validation ---
    contract_text = _validate_contract_text(contract_text)
    chosen_families = _validate_families(families)
    chosen_model = _validate_model(model)

    # --- Enqueue ---
    job_id = str(uuid.uuid4())
    job = Job(
        job_id=job_id,
        contract_id=contract_id,
        families=chosen_families,
        model=chosen_model,
    )
    _jobs[job_id] = job
    logger.info(
        "job_id=%s submitted contract_id=%s chars=%d families=%s model=%s",
        job_id, contract_id, len(contract_text), chosen_families, chosen_model,
    )

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_review, job_id, contract_text)

    return SubmitResponse(
        job_id=job_id,
        contract_id=contract_id,
        families=chosen_families,
        model=chosen_model,
        status=JobStatus.pending,
    )


@app.get("/review/{job_id}", response_model=StatusResponse)
async def get_review(job_id: str) -> StatusResponse:
    """
    Poll job status.
    When `status == 'done'` the full `result` object is included.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")

    return StatusResponse(
        job_id=job.job_id,
        contract_id=job.contract_id,
        families=job.families,
        model=job.model,
        status=job.status,
        error=job.error,
        result=job.result if job.status == JobStatus.done else None,
    )


@app.get("/review/{job_id}/report", response_class=HTMLResponse)
async def get_report(job_id: str) -> HTMLResponse:
    """Return the rendered HTML report for a completed job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    if job.status != JobStatus.done:
        raise HTTPException(
            status_code=409,
            detail=f"Job is '{job.status.value}'. Report is only available when status is 'done'.",
        )
    if not job.html_path or not Path(job.html_path).exists():
        raise HTTPException(status_code=500, detail="Report file missing on disk.")
    return HTMLResponse(content=Path(job.html_path).read_text(encoding="utf-8"))


@app.post("/review/{job_id}/summarize")
async def summarize_review(job_id: str) -> dict[str, Any]:
    """
    Generate overall_summary and top_red_flags for a completed review job.

    Triggers one LLM call (intentionally separate from the main review pipeline
    to keep review latency low). Updates the stored job result in place so
    subsequent GET /review/{job_id} calls include the summary.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    if job.status != JobStatus.done:
        raise HTTPException(
            status_code=409,
            detail=f"Job is '{job.status.value}'. Summary only available when status is 'done'.",
        )

    output = ContractReviewOutput(**job.result)
    llm = ClaudeClient(model=job.model)

    loop = asyncio.get_event_loop()
    summary, overall_risk, red_flags = await loop.run_in_executor(
        _executor,
        lambda: summarize_contract(
            contract_id=job.contract_id,
            clause_cards=output.clause_cards,
            llm=llm,
        ),
    )

    job.result["overall_summary"] = summary
    job.result["top_red_flags"] = red_flags
    job.result["overall_risk_rating"] = overall_risk.value

    return {
        "job_id": job_id,
        "overall_summary": summary,
        "overall_risk_rating": overall_risk,
        "top_red_flags": red_flags,
    }


@app.get("/models")
async def list_models() -> dict[str, Any]:
    """List the Claude models available for review jobs."""
    return {"models": AVAILABLE_MODELS, "default": MODEL}


@app.get("/families")
async def list_families() -> dict[str, Any]:
    """List the clause families that can be requested in a review."""
    return {"families": FAMILY_CATALOGUE, "all": list(CLAUSE_FAMILIES)}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
