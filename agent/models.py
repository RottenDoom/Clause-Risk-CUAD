"""
agent/models.py

All Pydantic v2 schemas for the contract review pipeline.
Every other agent module imports from here — no dict-based outputs elsewhere.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class SimilarPrecedent(BaseModel):
    contract_id: str
    why_similar: str  # non-empty; enforced by precedent_retrieval.py


class ContrastingPrecedent(BaseModel):
    contract_id: str
    why_contrasting: str  # non-empty; enforced by precedent_retrieval.py


# ---------------------------------------------------------------------------
# Per-family structured interpretation models
# ---------------------------------------------------------------------------

class AssignmentInterpretation(BaseModel):
    assignment_restricted: bool
    consent_required: bool
    restricted_party: str  # "assignor" | "assignee" | "both" | "neither" | "unknown"
    exceptions_detected: List[str]
    confidence: float = Field(ge=0.0, le=1.0)


class ChangeOfControlInterpretation(BaseModel):
    trigger_on_control_change: bool
    termination_right: bool
    consent_required: bool
    acquirer_bound: bool
    exceptions_detected: List[str]
    confidence: float = Field(ge=0.0, le=1.0)


class TerminationInterpretation(BaseModel):
    termination_for_convenience: bool
    notice_period_days: Optional[int]
    cure_period_days: Optional[int]
    mutual_termination_right: bool
    grounds_detected: List[str]
    confidence: float = Field(ge=0.0, le=1.0)


class ExclusivityInterpretation(BaseModel):
    exclusivity_granted: bool
    scope: str  # "full" | "partial" | "none" | "unknown"
    geographic_limitation: bool
    non_compete_present: bool
    duration_mentioned: bool
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Card and top-level output
# ---------------------------------------------------------------------------

class ClauseCard(BaseModel):
    clause_family: str
    clause_found: bool
    extracted_clause_text: Optional[str]
    structured_interpretation: Optional[dict]  # serialized per-family model
    similar_precedents: List[SimilarPrecedent]
    contrasting_precedents: List[ContrastingPrecedent]
    llm_generated_risk_rating: Optional[RiskLevel]
    risk_rationale: Optional[str]
    confidence_uncertainty_notes: List[str]


class ContractReviewOutput(BaseModel):
    contract_id: str
    overall_summary: str
    overall_risk_rating: RiskLevel
    top_red_flags: List[str]
    clause_cards: List[ClauseCard]  # always exactly 4 items
