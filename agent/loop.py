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

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Any

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

    async def run_async(
        self,
        contract_text: str,
        contract_id: str,
        families: list[str] | None = None,
        stream_queue: Optional[Any] = None,
    ) -> ContractReviewOutput:
        """
        Parallel version of run(): processes all clause families concurrently.

        Each family runs in its own thread (via run_in_executor). When
        stream_queue is provided (an asyncio.Queue), each ClauseCard is pushed
        to it as soon as that family finishes — enabling SSE streaming to the
        frontend. A None sentinel is pushed when all families are done.

        Wall time ≈ slowest single family instead of sum of all families.
        The sync run() is kept for the CLI (scripts/run_review.py).
        """
        run_families = families if families else CLAUSE_FAMILIES
        for f in run_families:
            if f not in CLAUSE_FAMILIES:
                raise ValueError(
                    f"Unknown clause family: {f!r}. Valid choices: {CLAUSE_FAMILIES}"
                )

        logger.info(
            "contract_id=%s start families=%s (parallel)", contract_id, run_families
        )
        t_total = time.monotonic()
        event_loop = asyncio.get_running_loop()

        cards: list[ClauseCard] = []

        with ThreadPoolExecutor(max_workers=len(run_families)) as pool:
            future_to_family = {
                event_loop.run_in_executor(
                    pool, self._process_family, contract_text, family
                ): family
                for family in run_families
            }

            # asyncio.wait preserves the original future objects (unlike as_completed),
            # so we can map each completed future back to its family name.
            pending = set(future_to_family.keys())
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for fut in done:
                    family = future_to_family[fut]
                    try:
                        result = fut.result()
                    except Exception as exc:
                        logger.error(
                            "contract_id=%s family=%s unhandled exception: %s",
                            contract_id, family, exc, exc_info=True,
                        )
                        continue

                    if result.card is not None:
                        cards.append(result.card)
                        logger.info(
                            "contract_id=%s family=%s done risk=%s",
                            contract_id, family, result.card.llm_generated_risk_rating,
                        )
                        if stream_queue is not None:
                            await stream_queue.put(result.card)
                    else:
                        logger.error(
                            "contract_id=%s family=%s no card produced", contract_id, family
                        )

        output = self._aggregate(contract_id, cards)
        logger.info(
            "contract_id=%s all done (%.1fs) overall_risk=%s",
            contract_id, time.monotonic() - t_total, output.overall_risk_rating,
        )

        if stream_queue is not None:
            await stream_queue.put(None)  # sentinel: stream complete

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
            discovery_score=result.discovery_score,
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
        t0 = time.monotonic()

        logger.info("family=%s step=1/discover starting", family)
        result = self._step1_discover(contract_text, family)
        logger.info("family=%s step=1/discover done clause_found=%s score=%.3f (%.1fs)",
                    family, result.clause_found, result.discovery_score, time.monotonic() - t0)

        logger.info("family=%s step=2/retrieve starting", family)
        result = self._step2_retrieve(result)
        logger.info("family=%s step=2/retrieve done similar=%d contrasting=%d (%.1fs)",
                    family, len(result.similar_precedents), len(result.contrasting_precedents), time.monotonic() - t0)

        logger.info("family=%s step=3/interpret starting", family)
        result = self._step3_interpret(result)
        logger.info("family=%s step=3/interpret done (%.1fs)", family, time.monotonic() - t0)

        return result
