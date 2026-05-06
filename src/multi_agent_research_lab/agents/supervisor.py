"""Supervisor / router: decides which worker runs next."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

log = logging.getLogger(__name__)

ROUTE_RESEARCHER = "researcher"
ROUTE_ANALYST = "analyst"
ROUTE_WRITER = "writer"
ROUTE_DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop.

    Routing policy (deterministic rule-based):
    1. No research_notes            → researcher
    2. Has research but no analysis → analyst
    3. Has analysis but no answer   → writer
    4. Has final_answer             → done
    5. Max iterations exceeded      → done (safety fallback)
    6. Errors accumulated (≥3)      → done (error fallback)
    """

    name = "supervisor"

    def __init__(self) -> None:
        self._max_iterations = get_settings().max_iterations

    def run(self, state: ResearchState) -> ResearchState:
        """Determine next route and record it in state."""

        with trace_span("supervisor", {"iteration": state.iteration}) as span:
            next_route = self._decide(state)
            state.record_route(next_route)
            span["next_route"] = next_route

        state.add_trace_event("supervisor_route", {"route": next_route, "iteration": state.iteration})
        log.info("SupervisorAgent → %s (iteration %d)", next_route, state.iteration)
        return state

    def _decide(self, state: ResearchState) -> str:
        # Safety: exceeded max iterations
        if state.iteration >= self._max_iterations:
            log.warning("Max iterations (%d) reached — stopping.", self._max_iterations)
            state.errors.append(f"Stopped at max_iterations={self._max_iterations}")
            return ROUTE_DONE

        # Safety: too many errors accumulated
        if len(state.errors) >= 3:
            log.warning("Too many errors (%d) — stopping.", len(state.errors))
            return ROUTE_DONE

        # Primary routing logic
        if state.final_answer:
            return ROUTE_DONE
        if state.analysis_notes:
            return ROUTE_WRITER
        if state.research_notes:
            return ROUTE_ANALYST
        return ROUTE_RESEARCHER
