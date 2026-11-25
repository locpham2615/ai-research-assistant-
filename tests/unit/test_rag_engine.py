from typing import Any

from ai_research_assistant.rag.context_builder import ContextBuilder
from ai_research_assistant.rag.rag_engine import (
    CitationGroundedRAG,
    verify_and_extract_citations,
)
from ai_research_assistant.retrieval.reranker import CrossEncoderReranker
from ai_research_assistant.retrieval.schemas import (
    RAGAnswerDraft,
    RetrievalTrace,
    RetrievedChunk,
    RewrittenQuery,
)


def _chunk(idx: int, section: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c_{idx}",
        doc_id="arxiv_1706.03762",
        topic_id="transformers",
        title="Attention Is All You Need",
        section=section,
        chunk_index=idx,
        text=text,
        rrf_score=0.03 - idx * 0.001,
    )


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    async def retrieve(
        self, query: str, **kwargs: Any
    ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        trace = RetrievalTrace(
            rewritten_query=RewrittenQuery(
                original_query=query,
                search_query="scaled dot-product attention sqrt d_k",
                bm25_keywords=["scaled dot-product", "sqrt"],
            ),
            dense_candidates=len(self.chunks),
            bm25_candidates=len(self.chunks),
            fused_candidates=len(self.chunks),
        )
        return self.chunks, trace


class FakeRAGGuard:
    def __init__(self, draft: RAGAnswerDraft) -> None:
        self.draft = draft
        self.calls = 0

    async def complete_structured(self, **kwargs: Any) -> RAGAnswerDraft:
        self.calls += 1
        return self.draft


def test_context_builder_enforces_max_chars_budget() -> None:
    chunks = [
        _chunk(0, "Abstract", "A" * 120),
        _chunk(1, "Method", "B" * 120),
        _chunk(2, "Experiments", "C" * 120),
    ]
    builder = ContextBuilder(max_context_chars=260)
    built = builder.build(chunks)

    assert built.total_chars <= 260
    assert list(built.sources.keys()) == [1]
    assert "[1] Title: Attention Is All You Need | Section: Abstract" in built.formatted_context
    assert "[2]" not in built.formatted_context


def test_verify_and_extract_citations_strips_hallucinated_indices() -> None:
    sources = {
        1: _chunk(0, "Attention", "Dot products are divided by sqrt(d_k)."),
        2: _chunk(1, "Softmax", "Large dot products push softmax into tiny gradients."),
    }
    raw = "Scaling by sqrt(d_k) [1, 99] prevents tiny gradients [2] and speeds up training [7]."
    cleaned, citations, invalid = verify_and_extract_citations(raw, sources)

    assert cleaned == (
        "Scaling by sqrt(d_k) [1] prevents tiny gradients [2] and speeds up training."
    )
    assert [c.index for c in citations] == [1, 2]
    assert citations[0].section == "Attention"
    assert invalid == [99, 7]


async def test_rag_engine_returns_grounded_answer_with_verified_citations() -> None:
    chunks = [
        _chunk(0, "Scaled Dot-Product Attention", "We divide dot products by sqrt(d_k)."),
        _chunk(1, "Training", "We use Adam optimizer."),
    ]
    retriever = FakeRetriever(chunks)
    reranker = CrossEncoderReranker(
        top_k=2,
        min_score=0.0,
        score_fn=lambda q, docs: [4.5, 0.2],
    )
    guard = FakeRAGGuard(
        RAGAnswerDraft(
            answer="The model divides dot products by sqrt(d_k) [1] and ignores noise [42].",
            is_sufficient_evidence=True,
        )
    )
    rag = CitationGroundedRAG(
        retriever=retriever,  # type: ignore[arg-type]
        reranker=reranker,
        schema_guard=guard,  # type: ignore[arg-type]
    )

    result = await rag.answer("Why scale by sqrt(d_k)?", topic_id="transformers")

    assert result.is_grounded is True
    assert result.abstain_reason == ""
    assert result.invalid_citations_removed == [42]
    assert len(result.citations) == 1
    assert result.citations[0].index == 1
    assert result.citations[0].section == "Scaled Dot-Product Attention"
    assert "[42]" not in result.answer
    assert result.retrieval_trace is not None
    assert len(result.retrieval_trace.reranked_chunks) == 2


async def test_rag_engine_abstention_gates() -> None:
    chunks = [_chunk(0, "Attention", "We divide dot products by sqrt(d_k).")]
    retriever = FakeRetriever(chunks)

    # Gate 1: Reranker lọc hết -> abstention mà không gọi LLM
    strict_reranker = CrossEncoderReranker(
        top_k=2,
        min_score=0.0,
        score_fn=lambda q, docs: [-3.5],
    )
    guard_unused = FakeRAGGuard(
        RAGAnswerDraft(answer="Should not be called", is_sufficient_evidence=True)
    )
    rag_gate1 = CitationGroundedRAG(
        retriever=retriever,  # type: ignore[arg-type]
        reranker=strict_reranker,
        schema_guard=guard_unused,  # type: ignore[arg-type]
    )
    res1 = await rag_gate1.answer("What is the budget of Llama 3?")
    assert res1.is_grounded is False
    assert res1.abstain_reason == "no_relevant_evidence"
    assert guard_unused.calls == 0

    # Gate 2: LLM tự xác nhận thiếu bằng chứng
    pass_reranker = CrossEncoderReranker(
        top_k=2,
        min_score=0.0,
        score_fn=lambda q, docs: [2.0],
    )
    guard_abstain = FakeRAGGuard(
        RAGAnswerDraft(
            answer="Tài liệu không đề cập chi phí huấn luyện.",
            is_sufficient_evidence=False,
            abstain_reason="missing_training_cost",
        )
    )
    rag_gate2 = CitationGroundedRAG(
        retriever=retriever,  # type: ignore[arg-type]
        reranker=pass_reranker,
        schema_guard=guard_abstain,  # type: ignore[arg-type]
    )
    res2 = await rag_gate2.answer("How much did training cost?")
    assert res2.is_grounded is False
    assert res2.abstain_reason == "missing_training_cost"

    # Gate 3: LLM trả lời nhưng chỉ chứa citation ảo -> đánh dấu ungrounded
    guard_fake_cite = FakeRAGGuard(
        RAGAnswerDraft(
            answer="It cost 10 million dollars [99].",
            is_sufficient_evidence=True,
        )
    )
    rag_gate3 = CitationGroundedRAG(
        retriever=retriever,  # type: ignore[arg-type]
        reranker=pass_reranker,
        schema_guard=guard_fake_cite,  # type: ignore[arg-type]
    )
    res3 = await rag_gate3.answer("How much did training cost?")
    assert res3.is_grounded is False
    assert res3.abstain_reason == "no_valid_citations"
    assert res3.invalid_citations_removed == [99]
    assert res3.citations == []
