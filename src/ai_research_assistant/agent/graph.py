import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from ai_research_assistant.agent.arxiv_tool import ArxivDiscoveryTool
from ai_research_assistant.agent.evidence_pool import (
    EvidencePool,
    merge_evidence_interleaved,
    merge_pre_rerank_chunks,
)
from ai_research_assistant.agent.grader import EvidenceGrader
from ai_research_assistant.agent.planner import QueryPlanner
from ai_research_assistant.agent.schemas import (
    AgentNextAction,
    AgentStepTrace,
    QueryPlan,
    SubQuery,
)
from ai_research_assistant.agent.state import ResearchAgentState
from ai_research_assistant.harness.llm_router import TaskType
from ai_research_assistant.harness.schema_guard import SchemaGuard
from ai_research_assistant.rag.context_builder import ContextBuilder
from ai_research_assistant.rag.rag_engine import verify_and_extract_citations
from ai_research_assistant.retrieval.schemas import (
    GroundedAnswer,
    RAGAnswerDraft,
    RetrievalTrace,
    RetrievedChunk,
    RewrittenQuery,
)

_AGENT_SYNTHESIS_PROMPT = (
    "You are an academic research assistant synthesizing strictly from the "
    "provided numbered sources.\n"
    "Rules:\n"
    "1. Answer ONLY using facts explicitly supported by sources [1], [2], ...\n"
    "2. Respond in the same language as the user's question.\n"
    "3. When comparing multiple papers or methods, cover each side explicitly "
    "and attach inline citations like [1] or [1, 2] to every factual claim.\n"
    "4. Never cite a number that is not present in the provided sources.\n"
    "5. STRICT ABSTENTION: If the question asks about a specific model, paper, "
    "dataset, benchmark, metric, or premise that is NOT explicitly present in "
    "the provided sources (even if related papers appear), you MUST set "
    "is_sufficient_evidence to false and explain the exact missing premise in "
    "abstain_reason. Never extrapolate or answer a different question.\n"
    "Return valid JSON matching the requested schema."
)


def _build_retrieval_trace(state: ResearchAgentState) -> RetrievalTrace:
    query = state.get("query", "")
    plan: QueryPlan | None = state.get("plan")
    executed: list[SubQuery] = state.get("executed_queries", [])
    pre_rerank: list[RetrievedChunk] = state.get("pre_rerank_chunks", [])
    evidence: list[RetrievedChunk] = state.get("evidence_chunks", [])

    if executed:
        combined_search = " | ".join(sq.search_query for sq in executed)
        keywords: list[str] = []
        for sq in executed:
            for kw in sq.bm25_keywords:
                if kw not in keywords:
                    keywords.append(kw)
        used_fallback = bool(plan and plan.reasoning.startswith("fallback:"))
    else:
        combined_search = query.strip()
        keywords = []
        used_fallback = True

    return RetrievalTrace(
        rewritten_query=RewrittenQuery(
            original_query=query,
            search_query=combined_search,
            bm25_keywords=keywords[:12],
            used_fallback=used_fallback,
        ),
        dense_candidates=state.get("dense_candidates_total", 0),
        bm25_candidates=state.get("bm25_candidates_total", 0),
        fused_candidates=len(pre_rerank),
        reranked_chunks=list(evidence),
    )


def build_research_graph(
    planner: QueryPlanner,
    evidence_pool: EvidencePool,
    grader: EvidenceGrader,
    schema_guard: SchemaGuard,
    context_builder: ContextBuilder | None = None,
    arxiv_tool: ArxivDiscoveryTool | None = None,
    max_context_chars: int = 7500,
) -> Any:
    ctx_builder = context_builder or ContextBuilder(max_context_chars=max_context_chars)

    async def plan_node(state: ResearchAgentState) -> dict[str, Any]:
        start = time.perf_counter()
        query = state.get("query", "")
        chat_history = state.get("chat_history") or None
        plan = await planner.plan(query, chat_history=chat_history)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        steps = list(state.get("steps", []))
        steps.append(
            AgentStepTrace(
                node="plan",
                hop=0,
                latency_ms=elapsed_ms,
                detail=f"route={plan.route}, sub_queries={len(plan.sub_queries)}",
            )
        )
        return {
            "plan": plan,
            "pending_queries": list(plan.sub_queries),
            "llm_calls": state.get("llm_calls", 0) + 1,
            "steps": steps,
        }

    async def retrieve_node(state: ResearchAgentState) -> dict[str, Any]:
        start = time.perf_counter()
        pending = list(state.get("pending_queries", []))
        hop = state.get("hop_count", 0) + 1
        top_k = state.get("top_k", evidence_pool.merged_top_k)

        batch = await evidence_pool.retrieve_batch(
            sub_queries=pending,
            topic_id=state.get("topic_id"),
            doc_ids=state.get("doc_ids"),
        )

        executed = list(state.get("executed_queries", [])) + pending
        merged_pre = merge_pre_rerank_chunks(
            state.get("pre_rerank_chunks", []),
            batch.pre_rerank_chunks,
        )
        merged_by_sq = dict(state.get("chunks_by_sub_query", {}))
        for sq_id, chunks in batch.chunks_by_sub_query.items():
            merged_by_sq[sq_id] = chunks

        merged_evidence = merge_evidence_interleaved(merged_by_sq, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        steps = list(state.get("steps", []))
        steps.append(
            AgentStepTrace(
                node="retrieve",
                hop=hop,
                latency_ms=elapsed_ms,
                detail=(
                    f"queries={len(pending)}, pre={len(merged_pre)}, "
                    f"evidence={len(merged_evidence)}"
                ),
            )
        )
        return {
            "pending_queries": [],
            "executed_queries": executed,
            "pre_rerank_chunks": merged_pre,
            "chunks_by_sub_query": merged_by_sq,
            "evidence_chunks": merged_evidence,
            "dense_candidates_total": (
                state.get("dense_candidates_total", 0) + batch.dense_candidates_total
            ),
            "bm25_candidates_total": (
                state.get("bm25_candidates_total", 0) + batch.bm25_candidates_total
            ),
            "hop_count": hop,
            "steps": steps,
        }

    async def grade_node(state: ResearchAgentState) -> dict[str, Any]:
        start = time.perf_counter()
        hop = state.get("hop_count", 1)
        allow_arxiv = bool(state.get("allow_external_arxiv", False)) and (
            arxiv_tool is not None
        )

        decision = await grader.grade(
            query=state.get("query", ""),
            plan=state.get("plan"),
            evidence_chunks=state.get("evidence_chunks", []),
            executed_queries=state.get("executed_queries", []),
            hop_count=hop,
            max_hops=state.get("max_hops", 2),
            allow_external_arxiv=allow_arxiv,
            used_external_arxiv=bool(state.get("used_external_arxiv", False)),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        next_action: AgentNextAction = decision.next_action
        pending = (
            list(decision.grade.follow_up_queries)
            if next_action == "refine"
            else []
        )
        grades = list(state.get("grades", [])) + [decision.grade]
        steps = list(state.get("steps", []))
        steps.append(
            AgentStepTrace(
                node="grade",
                hop=hop,
                latency_ms=elapsed_ms,
                detail=(
                    f"action={next_action}, fast_path={decision.used_fast_path}, "
                    f"follow_ups={len(pending)}"
                ),
            )
        )
        return {
            "last_grade": decision.grade,
            "grades": grades,
            "used_fast_path": state.get("used_fast_path", False)
            or decision.used_fast_path,
            "next_action": next_action,
            "pending_queries": pending,
            "llm_calls": state.get("llm_calls", 0) + (1 if decision.llm_called else 0),
            "steps": steps,
        }

    async def arxiv_discovery_node(state: ResearchAgentState) -> dict[str, Any]:
        start = time.perf_counter()
        hop = state.get("hop_count", 1)
        plan: QueryPlan | None = state.get("plan")
        search_query = (
            plan.sub_queries[0].search_query
            if plan and plan.sub_queries
            else state.get("query", "")
        )
        topic_id = state.get("topic_id") or "default"

        ingested: list[str] = []
        note = "tool_unavailable"
        if arxiv_tool is not None:
            res = await arxiv_tool.discover_and_ingest(
                query=search_query,
                topic_id=topic_id,
            )
            ingested = list(res.ingested_doc_ids)
            note = res.note

        next_action: AgentNextAction = "refine" if ingested else "abstain"
        retry_queries: list[SubQuery] = []
        if ingested:
            base_queries = (
                plan.sub_queries
                if plan and plan.sub_queries
                else [SubQuery(sub_query_id="sq1", search_query=search_query)]
            )
            retry_queries = [
                sq.model_copy(update={"sub_query_id": f"arxiv_sq{idx}"})
                for idx, sq in enumerate(base_queries, start=1)
            ]

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        steps = list(state.get("steps", []))
        steps.append(
            AgentStepTrace(
                node="arxiv_discovery",
                hop=hop,
                latency_ms=elapsed_ms,
                detail=f"ingested={len(ingested)}, note={note}",
            )
        )
        return {
            "used_external_arxiv": True,
            "external_ingested_docs": ingested,
            "pending_queries": retry_queries,
            "next_action": next_action,
            "steps": steps,
        }

    async def abstain_node(state: ResearchAgentState) -> dict[str, Any]:
        start = time.perf_counter()
        hop = state.get("hop_count", 0)
        trace = _build_retrieval_trace(state)
        grounded = GroundedAnswer(
            query=state.get("query", ""),
            answer=(
                "Không tìm thấy bằng chứng đủ liên quan trong kho tài liệu "
                "để trả lời câu hỏi này."
            ),
            citations=[],
            is_grounded=False,
            abstain_reason="no_relevant_evidence",
            retrieval_trace=trace,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        steps = list(state.get("steps", []))
        steps.append(
            AgentStepTrace(
                node="abstain",
                hop=hop,
                latency_ms=elapsed_ms,
                detail="no_relevant_evidence",
            )
        )
        return {
            "grounded_answer": grounded,
            "steps": steps,
        }

    async def synthesize_node(state: ResearchAgentState) -> dict[str, Any]:
        start = time.perf_counter()
        hop = state.get("hop_count", 0)
        query = state.get("query", "")
        evidence = state.get("evidence_chunks", [])
        trace = _build_retrieval_trace(state)

        if not evidence:
            grounded = GroundedAnswer(
                query=query,
                answer=(
                    "Không tìm thấy bằng chứng đủ liên quan trong kho tài liệu "
                    "để trả lời câu hỏi này."
                ),
                citations=[],
                is_grounded=False,
                abstain_reason="no_relevant_evidence",
                retrieval_trace=trace,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            steps = list(state.get("steps", []))
            steps.append(
                AgentStepTrace(
                    node="synthesize",
                    hop=hop,
                    latency_ms=elapsed_ms,
                    detail="empty_evidence_abstain",
                )
            )
            return {"grounded_answer": grounded, "steps": steps}

        built = ctx_builder.build(evidence)
        user_prompt = (
            f"Question:\n{query.strip()}\n\n"
            f"Sources:\n{built.formatted_context}"
        )

        draft = await schema_guard.complete_structured(
            task_type=TaskType.RAG_ANSWER,
            messages=[
                {"role": "system", "content": _AGENT_SYNTHESIS_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_model=RAGAnswerDraft,
        )

        if not draft.is_sufficient_evidence:
            reason = draft.abstain_reason.strip() or "insufficient_evidence"
            grounded = GroundedAnswer(
                query=query,
                answer=(
                    draft.answer.strip()
                    or "Các tài liệu hiện có không đủ thông tin để kết luận."
                ),
                citations=[],
                is_grounded=False,
                abstain_reason=reason,
                retrieval_trace=trace,
            )
        else:
            cleaned, citations, invalid_removed = verify_and_extract_citations(
                draft.answer,
                built.sources,
            )
            if not citations:
                grounded = GroundedAnswer(
                    query=query,
                    answer=cleaned,
                    citations=[],
                    is_grounded=False,
                    abstain_reason="no_valid_citations",
                    invalid_citations_removed=invalid_removed,
                    retrieval_trace=trace,
                )
            else:
                grounded = GroundedAnswer(
                    query=query,
                    answer=cleaned,
                    citations=citations,
                    is_grounded=True,
                    invalid_citations_removed=invalid_removed,
                    retrieval_trace=trace,
                )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        steps = list(state.get("steps", []))
        steps.append(
            AgentStepTrace(
                node="synthesize",
                hop=hop,
                latency_ms=elapsed_ms,
                detail=(
                    f"grounded={grounded.is_grounded}, "
                    f"citations={len(grounded.citations)}"
                ),
            )
        )
        return {
            "grounded_answer": grounded,
            "llm_calls": state.get("llm_calls", 0) + 1,
            "steps": steps,
        }

    def route_after_grade(state: ResearchAgentState) -> str:
        action = state.get("next_action", "synthesize")
        if action in {"refine", "arxiv_discovery", "abstain", "synthesize"}:
            return action
        return "synthesize"

    def route_after_arxiv(state: ResearchAgentState) -> str:
        return "retrieve" if state.get("next_action") == "refine" else "abstain"

    builder = StateGraph(ResearchAgentState)
    builder.add_node("plan", plan_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade", grade_node)
    builder.add_node("arxiv_discovery", arxiv_discovery_node)
    builder.add_node("abstain", abstain_node)
    builder.add_node("synthesize", synthesize_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "refine": "retrieve",
            "arxiv_discovery": "arxiv_discovery",
            "abstain": "abstain",
            "synthesize": "synthesize",
        },
    )
    builder.add_conditional_edges(
        "arxiv_discovery",
        route_after_arxiv,
        {
            "retrieve": "retrieve",
            "abstain": "abstain",
        },
    )
    builder.add_edge("abstain", END)
    builder.add_edge("synthesize", END)
    return builder.compile()