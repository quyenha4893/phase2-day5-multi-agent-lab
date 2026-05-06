"""Researcher agent: gathers sources and synthesises research notes."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are a meticulous research assistant. Given a query and a list of source snippets, "
    "synthesise clear, factual research notes (300-500 words). Cite each source by its title. "
    "Focus on key facts, definitions, and state-of-the-art findings."
)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._search = SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        with trace_span("researcher", {"query": state.request.query}) as span:
            sources = self._search.search(
                query=state.request.query,
                max_results=state.request.max_sources,
            )
            state.sources = sources

            snippets = "\n\n".join(
                f"[{i+1}] **{s.title}** ({s.url})\n{s.snippet}"
                for i, s in enumerate(sources)
            )
            user_prompt = (
                f"Query: {state.request.query}\n\n"
                f"Sources:\n{snippets}\n\n"
                f"Write research notes for audience: {state.request.audience}."
            )
            response = self._llm.complete(_SYSTEM, user_prompt, temperature=0.2)
            state.research_notes = response.content

            result = AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "sources_count": len(sources),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
            state.agent_results.append(result)
            span["input_tokens"] = response.input_tokens
            span["output_tokens"] = response.output_tokens

        state.add_trace_event("researcher_done", {"notes_length": len(state.research_notes or "")})
        log.info("ResearcherAgent done: %d sources, %d chars of notes", len(sources), len(state.research_notes or ""))
        return state
