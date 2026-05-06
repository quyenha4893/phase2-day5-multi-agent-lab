from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.agents.supervisor import ROUTE_ANALYST, ROUTE_RESEARCHER, ROUTE_WRITER, ROUTE_DONE
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_to_researcher_when_no_notes() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == ROUTE_RESEARCHER


def test_supervisor_routes_to_analyst_after_research() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.research_notes = "Some research notes here."
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == ROUTE_ANALYST


def test_supervisor_routes_to_writer_after_analysis() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.research_notes = "Research notes."
    state.analysis_notes = "Analysis notes."
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == ROUTE_WRITER


def test_supervisor_routes_done_when_answer_exists() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.research_notes = "Research."
    state.analysis_notes = "Analysis."
    state.final_answer = "Final answer."
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == ROUTE_DONE


def test_supervisor_stops_at_max_iterations() -> None:
    from multi_agent_research_lab.core.config import get_settings
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    max_iter = get_settings().max_iterations
    state.iteration = max_iter  # already at limit
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == ROUTE_DONE
    assert len(result.errors) > 0
