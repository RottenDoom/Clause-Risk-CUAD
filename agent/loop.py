"""
agent/loop.py

Main agentic loop. Runs the four-step review pipeline for a single contract
with explicit retry logic at each failure point.

Steps:
  1. Clause Discovery  — find clause span in raw contract text (local embeddings)
  2. Precedent Retrieval — similar + contrasting from ChromaDB
  3. Risk Interpretation — LLM → structured ClauseCard (with validation retry)
  4. Aggregation — overall summary + risk + red flags

All LLM calls go through the injected LLMClient; the loop itself never imports
anthropic. This makes the loop testable with mock clients and provider-agnostic.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from agent.models import ClauseCard, ContractReviewOutput
from config import CLAUSE_FAMILIES
from services.generation.base import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Carries state between steps for a single clause family."""
    family: str
    clause_found: bool = False
    clause_text: Optional[str] = None
    discovery_score: float = 0.0
    similar_precedents: list = field(default_factory=list)
    contrasting_precedents: list = field(default_factory=list)
    card: Optional[ClauseCard] = None


class ReviewLoop:
    """
    Orchestrates the four-step agentic review pipeline.

    Inject LLMClient and a configured Retriever so that the loop has no
    hard dependency on any specific provider or database implementation.

    Usage:
        from services.generation.claude_client import ClaudeClient
        from services.retrieval.retriever import Retriever

        loop = ReviewLoop(llm_client=ClaudeClient(), retriever=Retriever())
        output = loop.run(contract_text, contract_id="MyContract")
    """

    def __init__(self, llm_client: LLMClient, retriever) -> None:
        self.llm = llm_client
        self.retriever = retriever

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        contract_text: str,
        contract_id: str,
        families: list[str] | None = None,
    ) -> ContractReviewOutput:
        """
        Run the full review pipeline on a single contract.

        Args:
            families: Subset of CLAUSE_FAMILIES to process. None or empty means all 4.
                      Unknown family names raise ValueError immediately.

        Returns a ContractReviewOutput with one ClauseCard per requested family.
        """
        run_families = families if families else CLAUSE_FAMILIES
        for f in run_families:
            if f not in CLAUSE_FAMILIES:
                raise ValueError(
                    f"Unknown clause family: {f!r}. Valid choices: {CLAUSE_FAMILIES}"
                )

        logger.info("contract_id=%s start families=%s", contract_id, run_families)
        t_total = time.monotonic()

        cards: list[ClauseCard] = []
        for family in run_families:
            t_fam = time.monotonic()
            logger.info("contract_id=%s family=%s start", contract_id, family)
            result = self._process_family(contract_text, family)
            elapsed = time.monotonic() - t_fam
            if result.card is not None:
                cards.append(result.card)
                logger.info(
                    "contract_id=%s family=%s done (%.1fs) risk=%s",
                    contract_id, family, elapsed,
                    result.card.llm_generated_risk_rating,
                )
            else:
                logger.error("contract_id=%s family=%s no card produced (%.1fs)", contract_id, family, elapsed)

        output = self._aggregate(contract_id, cards)
        logger.info(
            "contract_id=%s all done (%.1fs) overall_risk=%s",
            contract_id, time.monotonic() - t_total, output.overall_risk_rating,
        )
        return output

    # ------------------------------------------------------------------
    # Step 1 — Clause Discovery
    # ------------------------------------------------------------------

    def _step1_discover(self, contract_text: str, family: str) -> StepResult:
        """
        Find the clause span in the raw contract using local embeddings.

        Retry strategy:
          - Attempt 1: standard ANCHOR_QUERIES (5 queries per family)
          - If score < DISCOVERY_MIN_SCORE: retry with broad_anchors
            (extended query set, threshold relaxed by 15%)
          - If still below threshold: clause_found=False (TODO: Failure handling)
        """
        # Import deferred so this module can be imported without sentence-transformers
        from agent.clause_discovery import discover_clause

        found, text, score = discover_clause(contract_text, family)

        if not found:
            logger.debug(
                "Family %s: score %.3f below threshold, retrying with broad anchors",
                family, score,
            )
            found, text, score = discover_clause(
                contract_text,
                family,
                broad=True,  # signals clause_discovery to use extended anchor set
            )

        if not found:
            logger.info("Family %s: clause not found after retry (score=%.3f)", family, score)

        return StepResult(
            family=family,
            clause_found=found,
            clause_text=text,
            discovery_score=score,
        )

    # ------------------------------------------------------------------
    # Step 2 — Precedent Retrieval
    # ------------------------------------------------------------------

    def _step2_retrieve(self, result: StepResult) -> StepResult:
        """
        Retrieve similar and contrasting precedents from ChromaDB via Retriever.

        Skipped entirely (returns empty lists) when clause_found=False —
        there is nothing to match against the reference index.
        No retry: retrieval is deterministic given the same embedding.
        """
        if not result.clause_found or result.clause_text is None:
            return result

        from agent.precedent_retrieval import retrieve_precedents

        similar, contrasting = retrieve_precedents(
            result.clause_text,
            result.family,
            self.retriever,
            self.llm,
        )
        result.similar_precedents = similar
        result.contrasting_precedents = contrasting
        return result

    # ------------------------------------------------------------------
    # Step 3 — Risk Interpretation
    # ------------------------------------------------------------------

    def _step3_interpret(self, result: StepResult) -> StepResult:
        """
        Interpret the clause and call the LLM to produce a structured ClauseCard.

        Retry is handled inside generate_risk_card (two attempts on JSON failure;
        graceful null degradation on second failure).
        """
        from agent.risk_rating import generate_risk_card

        card = generate_risk_card(
            family=result.family,
            clause_text=result.clause_text,
            clause_found=result.clause_found,
            similar=result.similar_precedents,
            contrasting=result.contrasting_precedents,
            llm=self.llm,
        )
        result.card = card
        return result

    # ------------------------------------------------------------------
    # Step 4 — Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self, contract_id: str, cards: list[ClauseCard]
    ) -> ContractReviewOutput:
        """
        Compute overall risk using rule-based logic only.

        Rule: any high card → high; any medium card (or None) → medium; else low.
        overall_summary and top_red_flags are left empty — call POST /review/{id}/summarize
        to populate them on demand.
        """
        from agent.summarizer import aggregate_risk

        overall_risk, null_flags = aggregate_risk(cards)
        return ContractReviewOutput(
            contract_id=contract_id,
            overall_risk_rating=overall_risk,
            top_red_flags=null_flags,
            clause_cards=cards,
        )

    # ------------------------------------------------------------------
    # Internal orchestration
    # ------------------------------------------------------------------

    def _process_family(self, contract_text: str, family: str) -> StepResult:
        """Run Steps 1->3 for a single clause family."""
        result = self._step1_discover(contract_text, family)
        result = self._step2_retrieve(result)
        result = self._step3_interpret(result)
        return result
