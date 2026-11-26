from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ai_research_assistant.agent.arxiv_tool import ArxivDiscoveryTool
from ai_research_assistant.agent.grader import EvidenceGrader
from ai_research_assistant.agent.schemas import EvidenceGrade, QueryPlan, SubQuery
from ai_research_assistant.retrieval.schemas import RetrievedChunk


def _chunk(chunk_id: str, doc_id: str, rerank_score: float, text: str = "Body") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        topic_id="llm_eval_v1",
        title=doc_id,
        section="Method",
        chunk_index=0,
        text=text,
        rrf_score=0.03,
        rerank_score=rerank_score,
    )


class FakeGraderGuard:
    def __init__(self, grade: EvidenceGrade | None = None, fail: bool = False) -> None:
        self.grade = grade
        self.fail = fail
        self.calls = 0

    async def complete_structured(self, **kwargs: Any) -> EvidenceGrade:
        self.calls += 1
        if self.fail or self.grade is None:
            raise RuntimeError("Grader LLM failure")
        return self.grade


async def test_grader_zero_evidence_fast_abstain_and_arxiv_gate() -> None:
    guard = FakeGraderGuard()
    grader = EvidenceGrader(schema_guard=guard)  # type: ignore[arg-type]

    # Mặc định allow_external_arxiv=False -> abstain ngay, 0 lần gọi LLM
    d1 = await grader.grade(
        query="What is Llama 3 budget?",
        plan=None,
        evidence_chunks=[],
        executed_queries=[],
        hop_count=1,
        max_hops=2,
        allow_external_arxiv=False,
    )
    assert d1.next_action == "abstain"
    assert d1.llm_called is False
    assert guard.calls == 0

    # Bật allow_external_arxiv=True và chưa tìm ngoài -> kích hoạt arxiv_discovery
    d2 = await grader.grade(
        query="What is FlashAttention-3?",
        plan=None,
        evidence_chunks=[],
        executed_queries=[],
        hop_count=1,
        max_hops=2,
        allow_external_arxiv=True,
        used_external_arxiv=False,
    )
    assert d2.next_action == "arxiv_discovery"
    assert guard.calls == 0

    # Đã tìm ngoài 1 lần mà vẫn rỗng -> phải abstain để không lặp vô hạn
    d3 = await grader.grade(
        query="What is FlashAttention-3?",
        plan=None,
        evidence_chunks=[],
        executed_queries=[],
        hop_count=1,
        max_hops=2,
        allow_external_arxiv=True,
        used_external_arxiv=True,
    )
    assert d3.next_action == "abstain"


async def test_grader_fast_path_simple_only_and_max_hops_gate() -> None:
    guard = FakeGraderGuard(
        grade=EvidenceGrade(is_sufficient=True, rationale="Covered both papers.")
    )
    grader = EvidenceGrader(
        schema_guard=guard,  # type: ignore[arg-type]
        fast_path_min_score=2.0,
        fast_path_min_chunks=2,
    )
    strong_chunks = [
        _chunk("c1", "arxiv_2106.09685", 4.2, "LoRA W_q W_v"),
        _chunk("c2", "arxiv_2106.09685", 3.1, "LoRA rank r=4"),
    ]

    # route='simple' + 2 chunk >= 2.0 -> Fast Path, không gọi LLM
    simple_plan = QueryPlan(
        route="simple",
        sub_queries=[SubQuery(search_query="LoRA matrices")],
    )
    d_fast = await grader.grade(
        query="Which matrices does LoRA adapt?",
        plan=simple_plan,
        evidence_chunks=strong_chunks,
        executed_queries=simple_plan.sub_queries,
        hop_count=1,
        max_hops=2,
    )
    assert d_fast.next_action == "synthesize"
    assert d_fast.used_fast_path is True
    assert d_fast.llm_called is False
    assert guard.calls == 0

    # route='decompose' -> KHÔNG đi Fast Path dù điểm cao (vì có thể thiếu bài thứ 2)
    decompose_plan = QueryPlan(
        route="decompose",
        sub_queries=[
            SubQuery(sub_query_id="sq1", search_query="LoRA matrices"),
            SubQuery(sub_query_id="sq2", search_query="QLoRA matrices"),
        ],
    )
    d_decomp = await grader.grade(
        query="Compare LoRA and QLoRA matrices",
        plan=decompose_plan,
        evidence_chunks=strong_chunks,
        executed_queries=decompose_plan.sub_queries,
        hop_count=1,
        max_hops=2,
    )
    assert d_decomp.used_fast_path is False
    assert d_decomp.llm_called is True
    assert guard.calls == 1

    # Chạm max_hops -> đi thẳng synthesize, không gọi thêm LLM
    d_cap = await grader.grade(
        query="Compare LoRA and QLoRA matrices",
        plan=decompose_plan,
        evidence_chunks=strong_chunks,
        executed_queries=decompose_plan.sub_queries,
        hop_count=2,
        max_hops=2,
    )
    assert d_cap.next_action == "synthesize"
    assert d_cap.llm_called is False
    assert guard.calls == 1


async def test_grader_refine_deduplicates_follow_ups_and_handles_llm_error() -> None:
    executed = [SubQuery(sub_query_id="sq1", search_query="ReAct Wikipedia API")]
    grade_with_dup = EvidenceGrade(
        is_sufficient=False,
        missing_aspects=["action types"],
        follow_up_queries=[
            SubQuery(search_query="  react wikipedia api  "),
            SubQuery(
                search_query="ReAct HotpotQA FEVER search lookup finish actions",
                bm25_keywords=["search", "lookup", "finish"],
            ),
        ],
    )
    grader = EvidenceGrader(
        schema_guard=FakeGraderGuard(grade=grade_with_dup),  # type: ignore[arg-type]
    )
    decision = await grader.grade(
        query="What three action types does ReAct define?",
        plan=QueryPlan(route="simple", sub_queries=executed),
        evidence_chunks=[_chunk("c1", "arxiv_2210.03629", 0.5, "Borderline score")],
        executed_queries=executed,
        hop_count=1,
        max_hops=2,
    )
    assert decision.next_action == "refine"
    assert len(decision.grade.follow_up_queries) == 1
    assert decision.grade.follow_up_queries[0].sub_query_id == "h2_sq1"

    # Nếu toàn bộ follow-up trùng truy vấn cũ -> chuyển sang synthesize thay vì lặp thừa
    only_dup_grader = EvidenceGrader(
        schema_guard=FakeGraderGuard(
            grade=EvidenceGrade(
                is_sufficient=False,
                follow_up_queries=[SubQuery(search_query="ReAct Wikipedia API")],
            )
        ),  # type: ignore[arg-type]
    )
    d_dup = await only_dup_grader.grade(
        query="What three action types does ReAct define?",
        plan=QueryPlan(route="simple", sub_queries=executed),
        evidence_chunks=[_chunk("c1", "arxiv_2210.03629", 0.5)],
        executed_queries=executed,
        hop_count=1,
        max_hops=2,
    )
    assert d_dup.next_action == "synthesize"

    # LLM lỗi -> fallback an toàn sang synthesize
    err_grader = EvidenceGrader(
        schema_guard=FakeGraderGuard(fail=True),  # type: ignore[arg-type]
    )
    d_err = await err_grader.grade(
        query="Any query",
        plan=None,
        evidence_chunks=[_chunk("c1", "doc1", 0.2)],
        executed_queries=executed,
        hop_count=1,
        max_hops=2,
    )
    assert d_err.next_action == "synthesize"
    assert d_err.grade.rationale.startswith("fallback:")


async def test_arxiv_discovery_tool_ingests_and_isolates_errors(tmp_path: Path) -> None:
    class FakeArxivClient:
        def __init__(self, fail: bool = False) -> None:
            self.fail = fail

        async def search(self, query: str, max_results: int = 2) -> list[Any]:
            if self.fail:
                raise ConnectionError("arXiv unreachable")
            return [
                SimpleNamespace(arxiv_id="2407.08608"),
                {"arxiv_id": "2309.06180"},
                SimpleNamespace(arxiv_id="2407.08608"),  # duplicate ignored
            ]

    class FakeIngestPipeline:
        async def ingest_arxiv(
            self,
            arxiv_id: str,
            topic_id: str,
            download_dir: Path,
            force: bool = False,
        ) -> Any:
            if arxiv_id == "2309.06180":
                return SimpleNamespace(status="failed", doc_id="arxiv_2309.06180")
            return SimpleNamespace(status="indexed", doc_id=f"arxiv_{arxiv_id}")

    tool = ArxivDiscoveryTool(
        arxiv_client=FakeArxivClient(),
        ingest_pipeline=FakeIngestPipeline(),
        download_dir=tmp_path / "pdfs",
        max_papers=2,
    )
    res = await tool.discover_and_ingest("FlashAttention-3", topic_id="llm_eval_v1")
    assert res.discovered_arxiv_ids == ["2407.08608", "2309.06180"]
    assert res.ingested_doc_ids == ["arxiv_2407.08608"]
    assert res.failed_arxiv_ids == ["2309.06180"]
    assert res.note == "ok"

    err_tool = ArxivDiscoveryTool(
        arxiv_client=FakeArxivClient(fail=True),
        ingest_pipeline=FakeIngestPipeline(),
        download_dir=tmp_path / "pdfs",
    )
    err_res = await err_tool.discover_and_ingest("FlashAttention-3", topic_id="llm_eval_v1")
    assert err_res.ingested_doc_ids == []
    assert err_res.note == "search_error:ConnectionError"