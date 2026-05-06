# Design Template

## Problem

Xây dựng hệ thống **research assistant** có thể nhận câu hỏi nghiên cứu phức tạp và trả lời bằng văn bản dài (~500 từ) có trích dẫn nguồn. Ví dụ: "Research GraphRAG state-of-the-art and write a 500-word summary."

## Why multi-agent?

Single-agent thực hiện một LLM call duy nhất → output bị giới hạn bởi những gì model "nhớ" từ lúc training, không có citation thật, không kiểm tra logic.

Multi-agent giải quyết bằng cách chia rõ trách nhiệm:
1. **Researcher** tìm và tổng hợp nguồn thực tế.
2. **Analyst** kiểm tra claim, so sánh viewpoints, flag weak evidence.
3. **Writer** viết final answer structured với citations.

→ Mỗi bước có thể được kiểm tra, retry độc lập; output cuối có chất lượng cao hơn.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Routing: quyết định agent nào chạy tiếp | `ResearchState` (toàn bộ) | `route_history` cập nhật | Max iterations exceeded → force DONE |
| Researcher | Tìm sources, synthesise research notes | `request.query`, `request.max_sources` | `state.sources`, `state.research_notes` | Search empty → fallback generic doc; LLM fail → retry 3× |
| Analyst | Phân tích notes, extract key claims, flag gaps | `research_notes` | `state.analysis_notes` | Missing research_notes → raise AgentExecutionError |
| Writer | Viết final answer với inline citations | `research_notes`, `analysis_notes`, `sources` | `state.final_answer` | Missing research_notes → raise AgentExecutionError |

## Shared state

`ResearchState` (Pydantic model) — single source of truth:

| Field | Type | Lý do cần |
|---|---|---|
| `request` | `ResearchQuery` | Chứa query gốc, max_sources, audience |
| `iteration` | `int` | Đếm số lần supervisor chạy; dùng cho max_iterations guard |
| `route_history` | `list[str]` | Audit trail: sequence of routing decisions |
| `sources` | `list[SourceDocument]` | Sources tìm được; Writer cần để cite |
| `research_notes` | `str \| None` | Output của Researcher; trigger analyst routing |
| `analysis_notes` | `str \| None` | Output của Analyst; trigger writer routing |
| `final_answer` | `str \| None` | Output của Writer; trigger DONE routing |
| `agent_results` | `list[AgentResult]` | Metadata (tokens, cost) per-agent cho benchmark |
| `trace` | `list[dict]` | Detailed event log cho observability |
| `errors` | `list[str]` | Accumulated errors; ≥3 triggers emergency DONE |

## Routing policy

```
START
  │
  ▼
[Supervisor] ──→ research_notes is None  ──→ [Researcher] ──┐
  │                                                           │
  │──→ analysis_notes is None            ──→ [Analyst]   ──┤
  │                                                           │
  │──→ final_answer is None              ──→ [Writer]    ──┤
  │                                                           │
  └──→ final_answer exists (or max_iter) ──→ END        ◄──┘
```

Implemented as LangGraph `StateGraph` with conditional edges from `supervisor` node.

## Guardrails

- **Max iterations**: 6 (configurable via `MAX_ITERATIONS` env var). Supervisor checks `state.iteration >= max_iterations` → force DONE.
- **Timeout**: 60 s per LLM call (set in `LLMClient.complete()` via `timeout=60`).
- **Retry**: tenacity `retry(stop=stop_after_attempt(3), wait=wait_exponential(...))` in `LLMClient`.
- **Fallback**: Agent exceptions are caught in workflow node wrappers → error appended to `state.errors`. If `len(errors) >= 3`, Supervisor stops.
- **Validation**: `ResearchQuery(query=query)` enforces `min_length=5`. `BenchmarkMetrics.quality_score` enforced `ge=0, le=10` by Pydantic.

## Benchmark plan

| Query | Metrics measured | Expected outcome |
|---|---|---|
| "Research GraphRAG state-of-the-art and write a 500-word summary" | Latency, cost, quality/10, citation_coverage | Multi-agent: higher quality (≥7), better citations; Baseline: faster (<5s) |
| "Compare single-agent and multi-agent workflows for customer support" | Latency, cost, quality/10 | Multi-agent: more balanced comparison; Baseline: may miss edge cases |
| "Summarize production guardrails for LLM agents" | Latency, cost, quality/10, citation_coverage | Multi-agent: structured list with sources; Baseline: shorter, less structured |
