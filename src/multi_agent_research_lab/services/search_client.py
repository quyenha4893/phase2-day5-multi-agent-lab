"""Search client abstraction for ResearcherAgent.

Uses a curated knowledge base mock when no external search API key is configured.
Falls back gracefully — agents should handle empty results without crashing.
"""

import logging

from multi_agent_research_lab.core.schemas import SourceDocument

log = logging.getLogger(__name__)

# Curated knowledge base: topic keywords → list of source documents
_KNOWLEDGE_BASE: list[dict] = [
    {
        "keywords": ["graphrag", "graph rag", "graph retrieval", "knowledge graph"],
        "title": "GraphRAG: Unlocking LLM discovery on narrative private data",
        "url": "https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/",
        "snippet": (
            "GraphRAG is a structured, hierarchical approach to RAG that uses LLMs to build a knowledge graph "
            "from private text corpora. It creates community summaries and enables global queries over large datasets. "
            "Unlike naive vector-search RAG, GraphRAG can answer questions that require synthesizing information "
            "from many disparate sources across a corpus."
        ),
    },
    {
        "keywords": ["graphrag", "graph rag", "graph retrieval"],
        "title": "From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
        "url": "https://arxiv.org/abs/2404.16130",
        "snippet": (
            "This paper presents Graph RAG, combining LLM-generated knowledge graphs with community detection "
            "and multi-stage summarization. Evaluation shows significant improvements on the comprehensiveness "
            "and diversity of answers compared to naive RAG baselines when answering global questions over large corpora."
        ),
    },
    {
        "keywords": ["multi-agent", "multi agent", "agent workflow", "agent systems"],
        "title": "Building Effective Agents — Anthropic Engineering",
        "url": "https://www.anthropic.com/engineering/building-effective-agents",
        "snippet": (
            "Anthropic recommends starting with the simplest system that works: a single LLM with tools. "
            "Multi-agent patterns are worth the complexity only when tasks are too long for one context window, "
            "or when independent verification is needed. Effective patterns include parallelisation, "
            "orchestrator-subagents, and evaluator-optimizer loops."
        ),
    },
    {
        "keywords": ["multi-agent", "multi agent", "agent workflow", "supervisor", "customer support"],
        "title": "OpenAI Agents SDK — Orchestration and Handoffs",
        "url": "https://platform.openai.com/docs/guides/agents/orchestration",
        "snippet": (
            "The OpenAI Agents SDK supports handoffs between agents: a triage agent can route queries to "
            "specialist agents (billing, technical support, etc.). Context is preserved through a shared "
            "thread. Key patterns: tool calls, structured outputs, and guardrails such as input/output validation."
        ),
    },
    {
        "keywords": ["langgraph", "lang graph", "state graph", "workflow graph", "agent graph"],
        "title": "LangGraph Concepts — State, Nodes, Edges",
        "url": "https://langchain-ai.github.io/langgraph/concepts/",
        "snippet": (
            "LangGraph models agent workflows as directed graphs where nodes are callables (agents/tools) "
            "and edges encode routing logic. Shared state flows through all nodes. Conditional edges allow "
            "dynamic routing based on state. Supports cycles, which enables iterative refinement loops. "
            "Compile the graph once; invoke it per query."
        ),
    },
    {
        "keywords": ["llm agent", "production", "guardrail", "safety", "production guardrail"],
        "title": "Production LLM Agent Guardrails — Best Practices",
        "url": "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "snippet": (
            "Production LLM agents require: (1) max iteration limits to prevent infinite loops, "
            "(2) timeout enforcement for each agent step, (3) input/output validation via schemas, "
            "(4) retry with exponential back-off on transient API errors, (5) fallback responses when "
            "agents fail, and (6) comprehensive logging and tracing for debugging."
        ),
    },
    {
        "keywords": ["rag", "retrieval augmented", "retrieval-augmented", "vector search", "embedding"],
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "url": "https://arxiv.org/abs/2005.11401",
        "snippet": (
            "RAG combines a neural retriever (dense passage retrieval) with a seq2seq generator. "
            "The retriever fetches relevant documents from a large corpus; the generator conditions "
            "on both the query and the retrieved documents. RAG-Token and RAG-Sequence variants offer "
            "different trade-offs between speed and quality."
        ),
    },
    {
        "keywords": ["benchmark", "evaluation", "llm evaluation", "quality metric"],
        "title": "Evaluating LLM Outputs: Metrics and Methodologies",
        "url": "https://huggingface.co/blog/llm-evaluation",
        "snippet": (
            "Evaluating LLMs requires multiple axes: factual accuracy (can be checked against ground truth), "
            "coherence (fluency and logical consistency), relevance (how well the response addresses the query), "
            "and citation coverage (proportion of claims backed by sources). Automated metrics include ROUGE, "
            "BERTScore, and LLM-as-judge rubrics; human evaluation remains the gold standard."
        ),
    },
]


class SearchClient:
    """Mock search client backed by a curated knowledge base.

    In production, swap this for Tavily, Bing Search API, or SerpAPI by
    implementing the same interface.
    """

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Return documents relevant to the query using keyword matching."""

        query_lower = query.lower()
        scored: list[tuple[int, dict]] = []
        for doc in _KNOWLEDGE_BASE:
            score = sum(1 for kw in doc["keywords"] if kw in query_lower)
            if score > 0:
                scored.append((score, doc))

        # sort by relevance score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[SourceDocument] = [
            SourceDocument(title=d["title"], url=d["url"], snippet=d["snippet"])
            for _, d in scored[:max_results]
        ]

        if not results:
            log.info("No keyword match for query '%s'; returning generic AI-agents overview.", query[:60])
            results = [
                SourceDocument(
                    title="LLM Powered Autonomous Agents — Lilian Weng",
                    url="https://lilianweng.github.io/posts/2023-06-23-agent/",
                    snippet=(
                        "A comprehensive overview of LLM-powered agents including components: "
                        "planning (task decomposition, reflection), memory (short/long-term), and "
                        "action (tool use, agent communication). Discusses challenges: finite context, "
                        "reliability of long-horizon planning, and alignment of sub-agents."
                    ),
                )
            ]

        log.info("SearchClient found %d result(s) for query '%s'", len(results), query[:60])
        return results
