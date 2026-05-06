"""Writer agent: produces the final answer with citations."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are an expert technical writer. Using the provided research notes and analysis, "
    "write a clear, well-structured response of approximately 500 words. "
    "Include inline citations using [Title] format. "
    "End with a 'Sources' section listing all referenced works. "
    "Adapt tone and depth to the specified audience."
)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        if not state.research_notes:
            raise AgentExecutionError("WriterAgent requires research_notes but none found in state")

        analysis_section = (
            f"\nAnalysis insights:\n{state.analysis_notes}" if state.analysis_notes else ""
        )
        sources_list = "\n".join(
            f"- {s.title}: {s.url}" for s in state.sources
        )

        with trace_span("writer", {}) as span:
            user_prompt = (
                f"Query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"Research notes:\n{state.research_notes}"
                f"{analysis_section}\n\n"
                f"Available sources:\n{sources_list}\n\n"
                "Write the final answer now."
            )
            response = self._llm.complete(_SYSTEM, user_prompt, temperature=0.4)
            state.final_answer = response.content

            result = AgentResult(
                agent=AgentName.WRITER,
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

        state.add_trace_event("writer_done", {"answer_length": len(state.final_answer or "")})
        log.info("WriterAgent done: %d chars in final answer", len(state.final_answer or ""))
        return state
