"""Analyst agent: turns research notes into structured insights."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are a critical analyst. Given research notes, produce structured analysis with: "
    "1) Key claims (bullet list), "
    "2) Comparison of different viewpoints or approaches, "
    "3) Identified knowledge gaps or weak evidence, "
    "4) A concise 1-2 sentence verdict on the current state. "
    "Be precise; flag any claim that lacks a clear source."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        if not state.research_notes:
            raise AgentExecutionError("AnalystAgent requires research_notes but none found in state")

        with trace_span("analyst", {"notes_length": len(state.research_notes)}) as span:
            user_prompt = (
                f"Original query: {state.request.query}\n\n"
                f"Research notes:\n{state.research_notes}\n\n"
                "Produce the structured analysis now."
            )
            response = self._llm.complete(_SYSTEM, user_prompt, temperature=0.1)
            state.analysis_notes = response.content

            result = AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
            state.agent_results.append(result)
            span["input_tokens"] = response.input_tokens
            span["output_tokens"] = response.output_tokens

        state.add_trace_event("analyst_done", {"analysis_length": len(state.analysis_notes or "")})
        log.info("AnalystAgent done: %d chars of analysis", len(state.analysis_notes or ""))
        return state
