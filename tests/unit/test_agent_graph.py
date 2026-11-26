from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ai_research_assistant.agent.arxiv_tool import ArxivDiscoveryTool
from ai_research_assistant.agent.evidence_pool import EvidencePool
from ai_research_assistant.agent.grader import EvidenceGrader
from ai_research_assistant.agent.planner import QueryPlanner
from ai_research_assistant.agent.research_agent import ResearchAgent
from ai_research_assistant.agent.schemas import EvidenceGrade, QueryPlan, SubQuery
from ai_research_assistant.retrieval.reranker import CrossEncoderReranker
from ai_research_assistant.retrieval.schemas import (
    RAGAnswerDraft,
    RetrievalTrace,
    RetrievedChunk,
    RewrittenQuery,
)


def _chunk(chunk_id: str, doc_id: str, text: str, rrf: float = 0.03) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        topic_id="llm_eval_v1",
        title=doc_id,
        section="Method",
        chunk_index=0,
        text=text,
        rrf_score=rrf,
    )


class ScriptedAgentGuard:
    def __init__(
        self,
        plan: QueryPlan,
        grade: EvidenceGrade | None = None,
        draft: RAGAnswerDraft | None = None,
    ) -> None:
        self.plan = plan
        self.grade = grade or EvidenceGrade(is_sufficient=True, rationale="Sufficient.")
        self.draft = draft or RAGAnswerDraft(
            answer="Supported by [1] and [2], ignoring fake [99].",
            is_sufficient_evidence=True,
        )
        self.task_calls: list[str] = []

    async def complete_structured(self, **kwargs: Any) -> Any:
        task = kwargs["task_type"]
        self.task_calls.append(str(task.value if hasattr(task, "value") else task))
        model = kwargs["response_model"]
        if model is QueryPlan:
            return self.plan
        if model is EvidenceGrade:
            return self.grade
        if model is RAGAnswerDraft:
            return self.draft
        raise AssertionError(f"Unexpected model: {model}")


class FakeRetriever:
    def __init__(self, corpus_fn: Any) -> None:
        self.corpus_fn = corpus_fn
        self.seen_queries: list[str] = []

    async def retrieve(
        self,
        query: str,
        topic_id: str | None = None,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
        rewritten_query: RewrittenQuery | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        self.seen_queries.append(query)
        chunks: list[RetrievedChunk] = self.corpus_fn(query)
        trace = RetrievalTrace(
            rewritten_query=rewritten_query
            or RewrittenQuery(original_query=query, search_query=query, bm25_keywords=[]),
            dense_candidates=len(chunks),
            bm25_candidates=len(chunks),
            fused_candidates=len(chunks),
        )
        return chunks, trace


async def test_agent_simple_fast_path_uses_two_llm_calls() -> None:
    guard = ScriptedAgentGuard(
        plan=QueryPlan(
            route="simple",
            sub_queries=[
                SubQuery(
                    sub_query_id="sq1",
                    search_query="scaled dot-product attention sqrt d_k",
                    bm25_keywords=["sqrt", "softmax"],
                )
            ],
        ),
        draft=RAGAnswerDraft(
            answer="Scaling by sqrt(d_k) keeps softmax gradients stable [1].",
            is_sufficient_evidence=True,
        ),
    )
    retriever = FakeRetriever(
        lambda q: [
            _chunk("c1", "arxiv_1706.03762", "Divide dot products by sqrt(d_k)."),
            _chunk("c2", "arxiv_1706.03762", "Large dot products cause small softmax gradients."),
        ]
    )
    reranker = CrossEncoderReranker(
        top_k=6,
        min_score=-3.0,
        score_fn=lambda q, docs: [4.5, 3.2],
    )
    agent = ResearchAgent(
        planner=QueryPlanner(schema_guard=guard),  # type: ignore[arg-type]
        evidence_pool=EvidencePool(retriever=retriever, reranker=reranker),
        grader=EvidenceGrader(schema_guard=guard),  # type: ignore[arg-type]
        schema_guard=guard,  # type: ignore[arg-type]
    )

    result = await agent.answer("Why scale by sqrt(d_k)?", topic_id="llm_eval_v1")

    assert result.grounded_answer.is_grounded is True
    assert result.agent_trace.route == "simple"
    assert result.agent_trace.hops_executed == 1
    assert result.agent_trace.used_fast_path is True
    assert result.agent_trace.llm_calls == 2
    assert guard.task_calls == ["agent_plan", "rag_answer"]
    assert len(result.pre_rerank_chunks) == 2


async def test_agent_decompose_self_correction_hop2_and_citation_verification() -> None:
    guard = ScriptedAgentGuard(
        plan=QueryPlan(
            route="decompose",
            sub_queries=[
                SubQuery(sub_query_id="sq1", search_query="LoRA target matrices"),
                SubQuery(sub_query_id="sq2", search_query="QLoRA adapters initial"),
            ],
        ),
        grade=EvidenceGrade(
            is_sufficient=False,
            missing_aspects=["QLoRA all linear blocks"],
            follow_up_queries=[
                SubQuery(
                    search_query="QLoRA all linear transformer layers NF4",
                    bm25_keywords=["QLoRA", "linear layers", "NF4"],
                )
            ],
        ),
        draft=RAGAnswerDraft(
            answer="LoRA adapts W_q and W_v [1], while QLoRA adapts all linear layers [2, 99].",
            is_sufficient_evidence=True,
        ),
    )

    def multi_hop_corpus(query: str) -> list[RetrievedChunk]:
        if "LoRA target" in query:
            return [_chunk("c_lora", "arxiv_2106.09685", "LoRA adapts W_q and W_v.")]
        if "all linear transformer" in query:
            return [_chunk("c_qlora", "arxiv_2305.14314", "QLoRA adapts all linear layers.")]
        return []

    retriever = FakeRetriever(multi_hop_corpus)
    reranker = CrossEncoderReranker(
        top_k=6,
        min_score=-3.0,
        score_fn=lambda q, docs: [4.0 for _ in docs],
    )
    agent = ResearchAgent(
        planner=QueryPlanner(schema_guard=guard),  # type: ignore[arg-type]
        evidence_pool=EvidencePool(retriever=retriever, reranker=reranker),
        grader=EvidenceGrader(schema_guard=guard),  # type: ignore[arg-type]
        schema_guard=guard,  # type: ignore[arg-type]
        max_hops=2,
    )

    result = await agent.answer("Compare LoRA and QLoRA target layers", topic_id="llm_eval_v1")

    assert result.agent_trace.route == "decompose"
    assert result.agent_trace.hops_executed == 2
    assert result.agent_trace.used_fast_path is False
    assert result.agent_trace.llm_calls == 3
    assert guard.task_calls == ["agent_plan", "agent_grade", "rag_answer"]
    assert result.grounded_answer.is_grounded is True
    assert result.grounded_answer.invalid_citations_removed == [99]
    assert "[99]" not in result.grounded_answer.answer
    cited_docs = {c.doc_id for c in result.grounded_answer.citations}
    assert cited_docs == {"arxiv_2106.09685", "arxiv_2305.14314"}


async def test_agent_zero_evidence_abstains_without_synthesis() -> None:
    guard = ScriptedAgentGuard(
        plan=QueryPlan(
            route="simple",
            sub_queries=[SubQuery(sub_query_id="sq1", search_query="Llama 3 budget")],
        )
    )
    retriever = FakeRetriever(
        lambda q: [_chunk("c_noise", "arxiv_1706.03762", "Unrelated text.")]
    )
    reranker = CrossEncoderReranker(
        top_k=6,
        min_score=-3.0,
        score_fn=lambda q, docs: [-5.5 for _ in docs],
    )
    agent = ResearchAgent(
        planner=QueryPlanner(schema_guard=guard),  # type: ignore[arg-type]
        evidence_pool=EvidencePool(retriever=retriever, reranker=reranker),
        grader=EvidenceGrader(schema_guard=guard),  # type: ignore[arg-type]
        schema_guard=guard,  # type: ignore[arg-type]
        allow_external_arxiv=False,
    )

    result = await agent.answer("What is the training budget of Llama 3?")
    assert result.grounded_answer.is_grounded is False
    assert result.grounded_answer.abstain_reason == "no_relevant_evidence"
    assert result.agent_trace.hops_executed == 1
    assert result.agent_trace.llm_calls == 1
    assert guard.task_calls == ["agent_plan"]


async def test_agent_external_arxiv_discovery_recovers_empty_store(tmp_path: Path) -> None:
    indexed_docs: list[str] = []

    class FakeArxivClient:
        async def search(self, query: str, max_results: int = 2) -> list[Any]:
            return [SimpleNamespace(arxiv_id="2407.08608")]

    class FakeIngestPipeline:
        async def ingest_arxiv(
            self,
            arxiv_id: str,
            topic_id: str,
            download_dir: Path,
            force: bool = False,
        ) -> Any:
            doc_id = f"arxiv_{arxiv_id}"
            indexed_docs.append(doc_id)
            return SimpleNamespace(status="indexed", doc_id=doc_id)

    def dynamic_corpus(query: str) -> list[RetrievedChunk]:
        if "arxiv_2407.08608" not in indexed_docs:
            return []
        return [
            _chunk(
                "c_fa3",
                "arxiv_2407.08608",
                "FlashAttention-3 uses Hopper TMA and FP8 quantization.",
            )
        ]

    guard = ScriptedAgentGuard(
        plan=QueryPlan(
            route="simple",
            sub_queries=[SubQuery(sub_query_id="sq1", search_query="FlashAttention-3 Hopper FP8")],
        ),
        draft=RAGAnswerDraft(
            answer="FlashAttention-3 leverages Hopper TMA and FP8 [1].",
            is_sufficient_evidence=True,
        ),
    )
    retriever = FakeRetriever(dynamic_corpus)
    reranker = CrossEncoderReranker(
        top_k=6,
        min_score=-3.0,
        score_fn=lambda q, docs: [4.2 for _ in docs],
    )
    arxiv_tool = ArxivDiscoveryTool(
        arxiv_client=FakeArxivClient(),
        ingest_pipeline=FakeIngestPipeline(),
        download_dir=tmp_path / "pdfs",
    )
    agent = ResearchAgent(
        planner=QueryPlanner(schema_guard=guard),  # type: ignore[arg-type]
        evidence_pool=EvidencePool(retriever=retriever, reranker=reranker),
        grader=EvidenceGrader(schema_guard=guard),  # type: ignore[arg-type]
        schema_guard=guard,  # type: ignore[arg-type]
        arxiv_tool=arxiv_tool,
        allow_external_arxiv=True,
        max_hops=2,
    )

    result = await agent.answer("How does FlashAttention-3 work?", topic_id="llm_eval_v1")

    assert result.agent_trace.used_external_arxiv is True
    assert result.agent_trace.hops_executed == 2
    assert result.grounded_answer.is_grounded is True
    assert result.grounded_answer.citations[0].doc_id == "arxiv_2407.08608"