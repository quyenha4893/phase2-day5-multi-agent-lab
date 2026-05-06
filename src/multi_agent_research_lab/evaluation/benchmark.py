"""Benchmark: single-agent vs multi-agent with cost, quality, and citation metrics."""

import logging
import re
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

log = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]

# Patterns that indicate a source citation in the final answer
_CITATION_PATTERNS = [re.compile(r"\[.+?\]"), re.compile(r"\(https?://")]


def _extract_cost(state: ResearchState) -> float:
    """Sum cost_usd from all agent_results metadata."""
    total = 0.0
    for result in state.agent_results:
        c = result.metadata.get("cost_usd")
        if c is not None:
            total += float(c)
    return total if total > 0 else 0.0


def _citation_coverage(state: ResearchState) -> float:
    """Estimate fraction of sentences in final_answer that contain a citation."""
    answer = state.final_answer or ""
    if not answer:
        return 0.0
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 20]
    if not sentences:
        return 0.0
    cited = sum(
        1
        for s in sentences
        if any(p.search(s) for p in _CITATION_PATTERNS)
    )
    return round(cited / len(sentences), 3)


def _llm_quality_score(query: str, answer: str) -> float:
    """Ask the LLM to rate the answer quality on a 0-10 rubric."""
    if not answer:
        return 0.0
    try:
        from multi_agent_research_lab.services.llm_client import LLMClient  # noqa: PLC0415

        llm = LLMClient()
        system = (
            "You are an impartial evaluator. Rate the following answer to the given query "
            "on a scale of 0.0 to 10.0 based on: accuracy, completeness, clarity, and citation quality. "
            "Reply with ONLY a single decimal number, e.g. '7.5'. No explanation."
        )
        user = f"Query: {query}\n\nAnswer:\n{answer[:2000]}"
        resp = llm.complete(system, user, temperature=0.0)
        score_str = resp.content.strip().split()[0]
        return min(10.0, max(0.0, float(score_str)))
    except Exception as exc:
        log.warning("Quality scoring failed: %s", exc)
        return 0.0


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, citation coverage, and LLM-judged quality."""

    started = perf_counter()
    failed = False
    state: ResearchState | None = None
    try:
        state = runner(query)
    except Exception as exc:
        log.error("Runner '%s' raised: %s", run_name, exc)
        failed = True

    latency = perf_counter() - started

    if failed or state is None:
        metrics = BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=latency,
            notes="FAILED",
        )
        from multi_agent_research_lab.core.schemas import ResearchQuery  # noqa: PLC0415
        state = ResearchState(request=ResearchQuery(query=query))
        return state, metrics

    cost = _extract_cost(state)
    citation_cov = _citation_coverage(state)
    quality = _llm_quality_score(query, state.final_answer or "")

    notes = (
        f"route={state.route_history} | "
        f"sources={len(state.sources)} | "
        f"citation_coverage={citation_cov:.0%} | "
        f"errors={len(state.errors)}"
    )

    failure_rate = 1.0 if len(state.errors) > 0 else 0.0

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 2),
        estimated_cost_usd=round(cost, 6) if cost else None,
        quality_score=round(quality, 1),
        citation_coverage=citation_cov,
        failure_rate=failure_rate,
        notes=notes,
    )
    log.info(
        "Benchmark '%s': latency=%.2fs cost=$%.5f quality=%.1f/10",
        run_name, latency, cost, quality,
    )
    return state, metrics
