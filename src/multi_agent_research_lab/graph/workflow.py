"""LangGraph workflow: Supervisor orchestrates Researcher → Analyst → Writer."""

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

log = logging.getLogger(__name__)


def _state_to_dict(state: ResearchState) -> dict[str, Any]:
    return state.model_dump()


def _dict_to_state(d: dict[str, Any]) -> ResearchState:
    return ResearchState.model_validate(d)


class MultiAgentWorkflow:
    """Builds and runs the Supervisor + Worker multi-agent graph via LangGraph."""

    def __init__(self) -> None:
        self._supervisor = SupervisorAgent()
        self._researcher = ResearcherAgent()
        self._analyst = AnalystAgent()
        self._writer = WriterAgent()

    def build(self) -> Any:
        """Create and compile a LangGraph StateGraph.

        Graph topology:
            START → supervisor
            supervisor --[researcher]--> researcher → supervisor
            supervisor --[analyst]----> analyst    → supervisor
            supervisor --[writer]-----> writer     → supervisor
            supervisor --[done]-------> END
        """

        graph: StateGraph = StateGraph(dict)  # use plain dict as LangGraph state

        # --- node wrappers (dict in → dict out) ---
        def supervisor_node(d: dict[str, Any]) -> dict[str, Any]:
            state = _dict_to_state(d)
            state = self._supervisor.run(state)
            return _state_to_dict(state)

        def researcher_node(d: dict[str, Any]) -> dict[str, Any]:
            state = _dict_to_state(d)
            try:
                state = self._researcher.run(state)
            except AgentExecutionError as exc:
                log.error("ResearcherAgent failed: %s", exc)
                state.errors.append(str(exc))
            return _state_to_dict(state)

        def analyst_node(d: dict[str, Any]) -> dict[str, Any]:
            state = _dict_to_state(d)
            try:
                state = self._analyst.run(state)
            except AgentExecutionError as exc:
                log.error("AnalystAgent failed: %s", exc)
                state.errors.append(str(exc))
            return _state_to_dict(state)

        def writer_node(d: dict[str, Any]) -> dict[str, Any]:
            state = _dict_to_state(d)
            try:
                state = self._writer.run(state)
            except AgentExecutionError as exc:
                log.error("WriterAgent failed: %s", exc)
                state.errors.append(str(exc))
            return _state_to_dict(state)

        # --- routing function (reads last entry in route_history) ---
        def route_from_supervisor(d: dict[str, Any]) -> str:
            history: list[str] = d.get("route_history", [])
            last = history[-1] if history else ROUTE_DONE
            if last == ROUTE_RESEARCHER:
                return "researcher"
            if last == ROUTE_ANALYST:
                return "analyst"
            if last == ROUTE_WRITER:
                return "writer"
            return END  # type: ignore[return-value]

        # --- register nodes ---
        graph.add_node("supervisor", supervisor_node)
        graph.add_node("researcher", researcher_node)
        graph.add_node("analyst", analyst_node)
        graph.add_node("writer", writer_node)

        # --- edges ---
        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            route_from_supervisor,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                END: END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Compile the graph, invoke it, and return the final ResearchState."""

        with trace_span("workflow", {"query": state.request.query}):
            compiled = self.build()
            initial_dict = _state_to_dict(state)
            log.info("Workflow starting for query: %s", state.request.query[:80])
            result_dict: dict[str, Any] = compiled.invoke(initial_dict)
            final_state = _dict_to_state(result_dict)

        log.info(
            "Workflow finished: %d iterations, route=%s",
            final_state.iteration,
            final_state.route_history,
        )
        return final_state
