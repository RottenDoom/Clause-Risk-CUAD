from pydantic import BaseModel
from typing import Any, Optional
from config import MODEL
from enum import Enum

# ---------------------------------------------------------------------------
# Job store
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class Job(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.pending
    contract_id: str = ""
    families: list[str] = []
    model: str = MODEL
    error: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    html_path: Optional[str] = None

# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------W

class SubmitResponse(BaseModel):
    job_id: str
    contract_id: str
    families: list[str]
    model: str
    status: JobStatus


class StatusResponse(BaseModel):
    job_id: str
    contract_id: str
    families: list[str]
    model: str
    status: JobStatus
    error: Optional[str] = None
    result: Optional[dict[str, Any]] = None