from pathlib import Path
from typing import Any

from qdrant_client.models import PointStruct

from ai_research_assistant.db.sqlite import SQLiteStore
from ai_research_assistant.db.vector_store import VectorStore
from ai_research_assistant.eval.dataset import audit_eval_dataset, load_eval_dataset
from ai_research_assistant.eval.judge import JudgeRubricOutput, RAGJudge
from ai_research_assistant.eval.reporter import (
    classify_failure,
    render_markdown_report,
    save_eval_report_json,
    save_markdown_report,
)
from ai_research_assistant.eval.runner import (
    DenseOnlyRetriever,
    EvalRunner,
    PassThroughReranker,
)
from ai_research_assistant.eval.schemas import (
    CorpusPaper,
    EvalDataset,
    EvalSample,
    SampleEvalResult,
)
from ai_research_assistant.pipelines.embedders.text_embedder import TextEmbedder
from ai_research_assistant.retrieval.reranker import CrossEncoderReranker
from ai_research_assistant.retrieval.schemas import (
    QueryRewriteOutput,
    RAGAnswerDraft,
    RetrievalTrace,
    RetrievedChunk,
    RewrittenQuery,
)


def test_real_golden_dataset_passes_audit() -> None:
    dataset_path = Path("evals/datasets/rag_golden_v1.json")
    dataset = load_eval_dataset(dataset_path)
    report = audit_eval_dataset(dataset)
    assert report.is_valid is True
    assert report.total_papers == 10
    assert report.total_samples == 60


class FakeMultiGuard:
    def __init__(self, fail_on_query: str | None = None) -> None:
        self.fail_on_query = fail_on_query

    async def complete_structured(self, **kwargs: Any) -> Any:
        response_model = kwargs["response_model"]
        user_content = str(kwargs["messages"][-1]["content"])

        if self.fail_on_query and self.fail_on_query in user_content:
            raise RuntimeError("Simulated LLM crash")

        if response_model is QueryRewriteOutput:
            return QueryRewriteOutput(
                search_query="scaled dot-product attention sqrt d_k",
                bm25_keywords=["dot products", "softmax", "gradients"],
            )
        if response_model is RAGAnswerDraft:
            return RAGAnswerDraft(
                answer="Dividing dot products by sqrt(d_k) prevents small softmax gradients [1].",
                is_sufficient_evidence=True,
            )
        if response_model is JudgeRubricOutput:
            return JudgeRubricOutput(
                faithfulness=5,
                correctness=5,
                relevancy=5,
                rationale="Exact match with evidence.",
            )
        raise AssertionError(f"Unexpected response_model: {response_model}")


async def test_dense_only_and_full_rag_runner_ablation_offline(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "eval.db")
    await store.init_db()

    vs = VectorStore(collection_name="eval_ablation", in_memory=True)
    vs.ensure_collection(vector_size=3)

    chunks = [
        {
            "chunk_id": "00000000-0000-0000-0000-000000000001",
            "doc_id": "arxiv_1706.03762v7",
            "topic_id": "llm_eval_v1",
            "title": "Attention Is All You Need",
            "section": "Scaled Dot-Product Attention",
            "chunk_index": 0,
            "text": "We scale dot products by 1/sqrt(d_k) to avoid small softmax gradients.",
            "text_hash": "h1",
            "vector": [1.0, 0.0, 0.0],
        },
        {
            "chunk_id": "00000000-0000-0000-0000-000000000002",
            "doc_id": "arxiv_2106.09685v2",
            "topic_id": "llm_eval_v1",
            "title": "LoRA",
            "section": "Method",
            "chunk_index": 0,
            "text": "LoRA freezes pretrained weights and injects rank decomposition matrices.",
            "text_hash": "h2",
            "vector": [0.0, 1.0, 0.0],
        },
    ]

    await store.upsert_chunks([{k: v for k, v in c.items() if k != "vector"} for c in chunks])
    vs.upsert_chunks(
        [
            PointStruct(
                id=c["chunk_id"],
                vector=c["vector"],
                payload={k: v for k, v in c.items() if k not in {"chunk_id", "vector"}},
            )
            for c in chunks
        ]
    )

    embedder = TextEmbedder(
        sqlite_store=store,
        embed_fn=lambda texts: [[1.0, 0.0, 0.0] for _ in texts],
    )
    guard = FakeMultiGuard()
    judge = RAGJudge(schema_guard=guard)  # type: ignore[arg-type]

    dataset = EvalDataset(
        dataset_name="mini_eval",
        topic_id="llm_eval_v1",
        corpus=[
            CorpusPaper(
                doc_id="arxiv_1706.03762",
                arxiv_id="1706.03762",
                title="Attention Is All You Need",
            ),
        ],
        samples=[
            EvalSample(
                sample_id="q001",
                question="Why scale dot products by sqrt(d_k)?",
                topic_id="llm_eval_v1",
                category="conceptual",
                should_abstain=False,
                expected_doc_ids=["arxiv_1706.03762"],
                expected_evidence_keywords=["dot products", "softmax", "gradients"],
                reference_answer="Prevents small softmax gradients.",
                evidence_quote="scale dot products by 1/sqrt(d_k) to avoid small softmax gradients",
            ),
            EvalSample(
                sample_id="q002",
                question="What is the training budget of Llama 3?",
                topic_id="llm_eval_v1",
                category="unanswerable",
                should_abstain=True,
                reference_answer="Not in corpus.",
            ),
        ],
    )

    # 1. dense_only: PassThroughReranker không lọc q002 -> leak trên unanswerable
    dense_runner = EvalRunner(
        mode="dense_only",
        retriever=DenseOnlyRetriever(embedder=embedder, vector_store=vs, retrieval_top_k=5),
        reranker=PassThroughReranker(top_k=2),
        schema_guard=guard,  # type: ignore[arg-type]
        judge=judge,
        retrieval_top_k=5,
        rerank_top_k=2,
    )
    progress_log: list[tuple[int, int, str]] = []
    dense_report = await dense_runner.run(
        dataset,
        on_sample_complete=lambda i, n, r: progress_log.append((i, n, r.sample_id)),
    )
    assert progress_log == [(1, 2, "q001"), (2, 2, "q002")]
    assert dense_report.retrieval.doc_recall_post_rerank == 1.0
    assert dense_report.abstention.false_negatives == 1
    assert dense_report.judge is not None
    assert dense_report.judge.mean_faithfulness == 1.0

    # 2. full_rag: CrossEncoderReranker lọc sạch q002 -> Abstention F1 = 1.0
    def fake_rerank_score(query: str, docs: list[str]) -> list[float]:
        if "Llama 3" in query:
            return [-3.0 for _ in docs]
        return [4.5 if "softmax" in d else -1.0 for d in docs]

    full_runner = EvalRunner(
        mode="full_rag",
        retriever=DenseOnlyRetriever(embedder=embedder, vector_store=vs, retrieval_top_k=5),
        reranker=CrossEncoderReranker(top_k=2, min_score=0.0, score_fn=fake_rerank_score),
        schema_guard=guard,  # type: ignore[arg-type]
        judge=judge,
        retrieval_top_k=5,
        rerank_top_k=2,
    )
    full_report = await full_runner.run(dataset)
    assert full_report.abstention.f1 == 1.0
    assert full_report.abstention.unanswerable_leak_rate == 0.0

    # 3. Kiểm tra Reporter xuất JSON + Markdown
    json_path = save_eval_report_json(full_report, tmp_path / "full_rag.json")
    md_path = save_markdown_report([dense_report, full_report], tmp_path / "summary.md")
    assert json_path.exists()
    md_text = md_path.read_text(encoding="utf-8")
    assert "## 1. Ablation Comparison" in md_text
    assert "`dense_only`" in md_text
    assert "`full_rag`" in md_text
    assert "unanswerable_leak" in md_text


async def test_runner_isolates_runtime_error_and_reporter_classifies_failures() -> None:
    class StaticRetriever:
        async def retrieve(
            self,
            query: str,
            topic_id: str | None = None,
            doc_ids: list[str] | None = None,
            top_k: int | None = None,
        ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
            chunk = RetrievedChunk(
                chunk_id="c1",
                doc_id="arxiv_1706.03762",
                topic_id="llm_eval_v1",
                title="Paper",
                section="Method",
                chunk_index=0,
                text="Some text",
                rrf_score=0.03,
            )
            trace = RetrievalTrace(
                rewritten_query=RewrittenQuery(
                    original_query=query,
                    search_query=query,
                    bm25_keywords=[],
                )
            )
            return [chunk], trace

    runner = EvalRunner(
        mode="hybrid_rrf",
        retriever=StaticRetriever(),
        reranker=PassThroughReranker(top_k=2),
        schema_guard=FakeMultiGuard(fail_on_query="CRASH_ME"),  # type: ignore[arg-type]
    )
    dataset = EvalDataset(
        dataset_name="err_eval",
        topic_id="llm_eval_v1",
        samples=[
            EvalSample(
                sample_id="q_err",
                question="CRASH_ME please",
                topic_id="llm_eval_v1",
                category="factoid",
                should_abstain=False,
                expected_doc_ids=["arxiv_1706.03762"],
                expected_evidence_keywords=["text"],
                reference_answer="Ref",
                evidence_quote="Some text",
            )
        ],
    )
    report = await runner.run(dataset)
    assert report.results[0].abstain_reason == "runtime_error:RuntimeError"
    assert classify_failure(report.results[0]) == "runtime_error:RuntimeError"

    over_filter = SampleEvalResult(
        sample_id="q_of",
        category="factoid",
        should_abstain=False,
        is_grounded=False,
        doc_recall_pre_rerank=1.0,
        doc_recall_post_rerank=0.0,
    )
    assert classify_failure(over_filter) == "false_refusal_rerank_over_filter"
    assert "## 5. Failure Analysis" in render_markdown_report([report])

async def test_eval_runner_agentic_mode_and_markdown_summary() -> None:
    from ai_research_assistant.agent.evidence_pool import EvidencePool
    from ai_research_assistant.agent.grader import EvidenceGrader
    from ai_research_assistant.agent.planner import QueryPlanner
    from ai_research_assistant.agent.research_agent import ResearchAgent
    from ai_research_assistant.agent.schemas import EvidenceGrade, QueryPlan, SubQuery

    class FakeAgentGuard:
        async def complete_structured(self, **kwargs: Any) -> Any:
            model = kwargs["response_model"]
            if model is QueryPlan:
                return QueryPlan(
                    route="decompose",
                    sub_queries=[
                        SubQuery(sub_query_id="sq1", search_query="LoRA target matrices"),
                        SubQuery(sub_query_id="sq2", search_query="QLoRA NF4 layers"),
                    ],
                )
            if model is EvidenceGrade:
                return EvidenceGrade(is_sufficient=True, rationale="Both papers covered.")
            if model is RAGAnswerDraft:
                return RAGAnswerDraft(
                    answer="LoRA adapts W_q/W_v [1], QLoRA adapts all linear layers [2].",
                    is_sufficient_evidence=True,
                )
            raise AssertionError(f"Unexpected model: {model}")

    class TwoPaperRetriever:
        async def retrieve(
            self,
            query: str,
            topic_id: str | None = None,
            doc_ids: list[str] | None = None,
            top_k: int | None = None,
            rewritten_query: RewrittenQuery | None = None,
        ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
            if "QLoRA" in query:
                chunk = RetrievedChunk(
                    chunk_id="c_qlora",
                    doc_id="arxiv_2305.14314",
                    topic_id="llm_eval_v1",
                    title="QLoRA",
                    section="Method",
                    chunk_index=0,
                    text="QLoRA adapts all linear transformer layers using NF4.",
                    rrf_score=0.03,
                )
            else:
                chunk = RetrievedChunk(
                    chunk_id="c_lora",
                    doc_id="arxiv_2106.09685",
                    topic_id="llm_eval_v1",
                    title="LoRA",
                    section="Method",
                    chunk_index=0,
                    text="LoRA adapts W_q and W_v attention matrices.",
                    rrf_score=0.03,
                )
            trace = RetrievalTrace(
                rewritten_query=rewritten_query
                or RewrittenQuery(original_query=query, search_query=query, bm25_keywords=[]),
                dense_candidates=1,
                bm25_candidates=1,
                fused_candidates=1,
            )
            return [chunk], trace

    guard = FakeAgentGuard()
    reranker = CrossEncoderReranker(
        top_k=6,
        min_score=-3.0,
        score_fn=lambda q, docs: [4.0 for _ in docs],
    )
    agent = ResearchAgent(
        planner=QueryPlanner(schema_guard=guard),  # type: ignore[arg-type]
        evidence_pool=EvidencePool(retriever=TwoPaperRetriever(), reranker=reranker),
        grader=EvidenceGrader(schema_guard=guard),  # type: ignore[arg-type]
        schema_guard=guard,  # type: ignore[arg-type]
    )
    runner = EvalRunner(mode="agentic_rag", agent=agent, retrieval_top_k=10, rerank_top_k=6)
    dataset = EvalDataset(
        dataset_name="agentic_mini",
        topic_id="llm_eval_v1",
        samples=[
            EvalSample(
                sample_id="q044",
                question="Compare LoRA and QLoRA target layers",
                topic_id="llm_eval_v1",
                category="multi_paper",
                should_abstain=False,
                expected_doc_ids=["arxiv_2106.09685", "arxiv_2305.14314"],
                expected_evidence_keywords=["attention matrices", "linear transformer"],
                reference_answer="LoRA adapts W_q/W_v while QLoRA adapts all linear layers.",
                evidence_quote="attention matrices || linear transformer",
            )
        ],
    )

    report = await runner.run(dataset)
    assert report.retrieval.doc_recall_post_rerank == 1.0
    assert report.retrieval.evidence_hit_post_rerank == 1.0
    assert report.agent is not None
    assert report.agent.decompose_count == 1
    assert report.agent.avg_hops == 1.0

    md = render_markdown_report([report])
    assert "## Agentic Workflow Summary" in md
    assert "`agentic_rag`" in md