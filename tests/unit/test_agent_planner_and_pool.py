from typing import Any

import pytest

from ai_research_assistant.agent.evidence_pool import (
    EvidencePool,
    merge_evidence_interleaved,
    merge_pre_rerank_chunks,
)
from ai_research_assistant.agent.planner import QueryPlanner
from ai_research_assistant.agent.schemas import QueryPlan, SubQuery
from ai_research_assistant.retrieval.reranker import CrossEncoderReranker
from ai_research_assistant.retrieval.schemas import (
    RetrievalTrace,
    RetrievedChunk,
    RewrittenQuery,
)


def _chunk(
    chunk_id: str,
    doc_id: str,
    text: str,
    rrf: float = 0.03,
    rerank: float | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        topic_id="llm_eval_v1",
        title=doc_id,
        section="Method",
        chunk_index=0,
        text=text,
        rrf_score=rrf,
        rerank_score=rerank,
    )


class FakePlannerGuard:
    def __init__(self, plan: QueryPlan | None = None, fail: bool = False) -> None:
        self.plan = plan
        self.fail = fail
        self.calls = 0

    async def complete_structured(self, **kwargs: Any) -> QueryPlan:
        self.calls += 1
        if self.fail or self.plan is None:
            raise RuntimeError("Planner LLM failure")
        return self.plan


async def test_query_planner_decompose_caps_and_fallback() -> None:
    expected = QueryPlan(
        route="decompose",
        sub_queries=[
            SubQuery(
                sub_query_id="sq1",
                search_query="LoRA target attention matrices GPT-3",
                bm25_keywords=["LoRA", "W_q", "W_v"],
                target_aspect="LoRA matrices",
            ),
            SubQuery(
                sub_query_id="sq2",
                search_query="QLoRA adapters all linear transformer blocks",
                bm25_keywords=["QLoRA", "NF4", "linear layers"],
                target_aspect="QLoRA matrices",
            ),
            SubQuery(
                sub_query_id="sq3",
                search_query="extra aspect",
                bm25_keywords=["extra"],
            ),
        ],
        reasoning="Compares LoRA and QLoRA.",
    )
    planner = QueryPlanner(
        schema_guard=FakePlannerGuard(plan=expected),  # type: ignore[arg-type]
        max_sub_queries=2,
    )
    capped = await planner.plan("So sánh LoRA và QLoRA về ma trận áp dụng adapter")
    assert capped.route == "decompose"
    assert [sq.sub_query_id for sq in capped.sub_queries] == ["sq1", "sq2"]

    fallback_planner = QueryPlanner(
        schema_guard=FakePlannerGuard(fail=True),  # type: ignore[arg-type]
    )
    fb = await fallback_planner.plan("Tại sao chia cho sqrt(d_k)?")
    assert fb.route == "simple"
    assert len(fb.sub_queries) == 1
    assert fb.sub_queries[0].search_query == "Tại sao chia cho sqrt(d_k)?"
    assert fb.reasoning.startswith("fallback:")

    with pytest.raises(ValueError, match="empty"):
        await planner.plan("   ")


def test_merge_evidence_interleaved_balances_multi_paper_despite_overlap() -> None:
    # sq1 (LoRA) có điểm rerank cao hơn hẳn sq2 (QLoRA) và 2 bên trùng nhau ở c_shared
    by_sq = {
        "sq1": [
            _chunk("c_shared", "arxiv_2106.09685", "LoRA overview", rerank=5.0),
            _chunk("c_lora_2", "arxiv_2106.09685", "LoRA W_q W_v", rerank=4.8),
            _chunk("c_lora_3", "arxiv_2106.09685", "LoRA rank r=4", rerank=4.5),
        ],
        "sq2": [
            _chunk("c_shared", "arxiv_2106.09685", "LoRA overview", rerank=6.2),
            _chunk("c_qlora_1", "arxiv_2305.14314", "QLoRA all linear layers", rerank=1.5),
            _chunk("c_qlora_2", "arxiv_2305.14314", "QLoRA NF4 double quant", rerank=1.2),
        ],
    }

    merged = merge_evidence_interleaved(by_sq, top_k=4)

    # Round 1: c_shared (từ sq1, giữ max rerank 6.2), c_qlora_1 (từ sq2 vì bỏ qua c_shared trùng)
    # Round 2: c_lora_2 (từ sq1), c_qlora_2 (từ sq2)
    assert [c.chunk_id for c in merged] == [
        "c_shared",
        "c_qlora_1",
        "c_lora_2",
        "c_qlora_2",
    ]
    assert merged[0].rerank_score == 6.2
    docs = {c.doc_id for c in merged}
    assert docs == {"arxiv_2106.09685", "arxiv_2305.14314"}


def test_merge_pre_rerank_chunks_deduplicates_and_keeps_higher_rrf() -> None:
    a = [_chunk("c1", "docA", "text1", rrf=0.020), _chunk("c2", "docA", "text2", rrf=0.015)]
    b = [_chunk("c1", "docA", "text1", rrf=0.035), _chunk("c3", "docB", "text3", rrf=0.025)]
    merged = merge_pre_rerank_chunks(a, b)
    assert [c.chunk_id for c in merged] == ["c1", "c3", "c2"]
    assert merged[0].rrf_score == 0.035


async def test_evidence_pool_passes_prebuilt_rewritten_query_and_reranks() -> None:
    captured_rewrites: list[RewrittenQuery | None] = []

    class FakeSubQueryRetriever:
        async def retrieve(
            self,
            query: str,
            topic_id: str | None = None,
            doc_ids: list[str] | None = None,
            top_k: int | None = None,
            rewritten_query: RewrittenQuery | None = None,
        ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
            captured_rewrites.append(rewritten_query)
            doc_id = "arxiv_2305.14314" if "QLoRA" in query else "arxiv_2106.09685"
            chunks = [
                _chunk(f"{doc_id}_0", doc_id, f"Best match for {query}", rrf=0.032),
                _chunk(f"{doc_id}_1", doc_id, "Secondary background", rrf=0.020),
            ]
            trace = RetrievalTrace(
                rewritten_query=rewritten_query
                or RewrittenQuery(original_query=query, search_query=query, bm25_keywords=[]),
                dense_candidates=2,
                bm25_candidates=1,
                fused_candidates=2,
            )
            return chunks, trace

    reranker = CrossEncoderReranker(
        top_k=4,
        min_score=-3.0,
        score_fn=lambda q, docs: [3.5 if "Best match" in d else -4.0 for d in docs],
    )
    pool = EvidencePool(
        retriever=FakeSubQueryRetriever(),
        reranker=reranker,
        per_sub_query_top_k=3,
        merged_top_k=6,
    )

    sub_queries = [
        SubQuery(sub_query_id="sq1", search_query="LoRA matrices", bm25_keywords=["LoRA"]),
        SubQuery(sub_query_id="sq2", search_query="QLoRA NF4", bm25_keywords=["QLoRA", "NF4"]),
    ]
    batch = await pool.retrieve_batch(sub_queries, topic_id="llm_eval_v1")

    assert len(captured_rewrites) == 2
    assert captured_rewrites[0] is not None
    assert captured_rewrites[0].bm25_keywords == ["LoRA"]
    assert captured_rewrites[1] is not None
    assert captured_rewrites[1].bm25_keywords == ["QLoRA", "NF4"]

    assert batch.dense_candidates_total == 4
    assert batch.bm25_candidates_total == 2
    assert len(batch.pre_rerank_chunks) == 4
    assert [c.doc_id for c in batch.chunks_by_sub_query["sq1"]] == ["arxiv_2106.09685"]
    assert [c.doc_id for c in batch.chunks_by_sub_query["sq2"]] == ["arxiv_2305.14314"]