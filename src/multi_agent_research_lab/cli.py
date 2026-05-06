"""Command-line entrypoint for the lab starter."""

import pathlib
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()

_BASELINE_SYSTEM = (
    "You are a knowledgeable research assistant. Answer the user's query with a clear, "
    "well-structured response of approximately 500 words. Include relevant facts, "
    "definitions, and current best practices. Cite sources where applicable."
)


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run single-agent baseline: one LLM call, no sub-agents."""

    _init()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    llm = LLMClient()
    response = llm.complete(_BASELINE_SYSTEM, query)
    state.final_answer = response.content
    state.add_trace_event("baseline_done", {
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost_usd": response.cost_usd,
    })

    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))
    console.print(
        f"[dim]tokens in={response.input_tokens} out={response.output_tokens} "
        f"cost=${response.cost_usd:.5f}[/dim]" if response.cost_usd else ""
    )


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow (Supervisor + Researcher + Analyst + Writer)."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc

    console.print(Panel.fit(result.final_answer or "(no answer)", title="Multi-Agent Answer"))
    console.print(f"[dim]route={result.route_history}  iterations={result.iteration}[/dim]")
    if result.errors:
        console.print(f"[yellow]Errors: {result.errors}[/yellow]")


@app.command()
def benchmark(
    output: Annotated[
        pathlib.Path,
        typer.Option("--output", "-o", help="Output path for benchmark report"),
    ] = pathlib.Path("reports/benchmark_report.md"),
) -> None:
    """Run baseline vs multi-agent on all benchmark queries and save a Markdown report."""

    _init()
    import yaml  # noqa: PLC0415

    config_path = pathlib.Path("configs/lab_default.yaml")
    queries: list[str] = []
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        queries = cfg.get("benchmark", {}).get("queries", [])
    if not queries:
        queries = ["Research GraphRAG state-of-the-art and write a 500-word summary"]

    llm = LLMClient()
    workflow = MultiAgentWorkflow()

    def baseline_runner(q: str) -> ResearchState:
        s = ResearchState(request=ResearchQuery(query=q))
        resp = llm.complete(_BASELINE_SYSTEM, q)
        s.final_answer = resp.content
        s.add_trace_event("baseline_done", {"cost_usd": resp.cost_usd})
        # stash cost in agent_results so benchmark can pick it up
        from multi_agent_research_lab.core.schemas import AgentName, AgentResult  # noqa: PLC0415
        s.agent_results.append(AgentResult(
            agent=AgentName.RESEARCHER,
            content=resp.content,
            metadata={"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens, "cost_usd": resp.cost_usd},
        ))
        return s

    def multi_runner(q: str) -> ResearchState:
        s = ResearchState(request=ResearchQuery(query=q))
        return workflow.run(s)

    all_metrics = []
    for i, query in enumerate(queries, 1):
        console.print(f"\n[bold]Query {i}/{len(queries)}:[/bold] {query[:70]}...")
        console.print("[cyan]  Running baseline...[/cyan]")
        _, b_metrics = run_benchmark(f"baseline_q{i}", query, baseline_runner)
        all_metrics.append(b_metrics)

        console.print("[cyan]  Running multi-agent...[/cyan]")
        _, m_metrics = run_benchmark(f"multi_agent_q{i}", query, multi_runner)
        all_metrics.append(m_metrics)

    report = render_markdown_report(all_metrics)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    console.print(Panel.fit(f"Report saved to [bold]{output}[/bold]", style="green"))
    console.print(report)


if __name__ == "__main__":
    app()
