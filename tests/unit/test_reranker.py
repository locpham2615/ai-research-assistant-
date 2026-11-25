from ai_research_assistant.retrieval.reranker import CrossEncoderReranker
from ai_research_assistant.retrieval.schemas import RetrievedChunk


def _make_chunk(
    chunk_id: str,
    section: str,
    text: str,
    rrf_score: float,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc_1",
        topic_id="transformers",
        title="Attention Is All You Need",
        section=section,
        chunk_index=0,
        text=text,
        dense_score=0.8,
        lexical_score=2.0,
        rrf_score=rrf_score,
    )


async def test_reranker_reorders_filters_and_slices_top_k() -> None:
    chunks = [
        _make_chunk("c1", "Intro", "General background.", rrf_score=0.032),
        _make_chunk("c2", "Attention", "Divide dot product by sqrt d_k.", rrf_score=0.025),
        _make_chunk("c3", "Softmax", "Large dot products push softmax gradients.", rrf_score=0.020),
        _make_chunk("c4", "Training", "Adam optimizer with beta1=0.9.", rrf_score=0.015),
    ]

    captured_docs: list[str] = []

    def fake_score(query: str, docs: list[str]) -> list[float]:
        captured_docs.extend(docs)
        return [0.4, 5.2, 3.1, -2.5]

    reranker = CrossEncoderReranker(top_k=2, min_score=0.0, score_fn=fake_score)
    results = await reranker.rerank("Why scale by sqrt d_k?", chunks)

    # Giữ top 2, sắp giảm dần theo rerank_score, loại c4 (-2.5 < 0.0)
    assert [c.chunk_id for c in results] == ["c2", "c3"]
    assert results[0].rerank_score == 5.2
    assert results[1].rerank_score == 3.1
    assert captured_docs[1].startswith("Attention: ")

    # Bảo toàn metadata gốc và không mutate danh sách input
    assert results[0].dense_score == 0.8
    assert results[0].lexical_score == 2.0
    assert chunks[0].rerank_score is None


async def test_reranker_returns_empty_when_all_below_threshold() -> None:
    chunks = [
        _make_chunk("c1", "Intro", "Unrelated text.", rrf_score=0.03),
        _make_chunk("c2", "Training", "Another unrelated text.", rrf_score=0.02),
    ]

    def fake_score(query: str, docs: list[str]) -> list[float]:
        return [-4.1, -1.2]

    reranker = CrossEncoderReranker(top_k=5, min_score=0.0, score_fn=fake_score)
    results = await reranker.rerank("What is the budget of Llama 3?", chunks)
    assert results == []


async def test_reranker_handles_empty_inputs_and_tie_breaks_by_rrf() -> None:
    chunks = [
        _make_chunk("c_low_rrf", "Method", "Text A", rrf_score=0.015),
        _make_chunk("c_high_rrf", "Method", "Text B", rrf_score=0.030),
    ]

    def tie_score(query: str, docs: list[str]) -> list[float]:
        return [2.0, 2.0]

    reranker = CrossEncoderReranker(top_k=5, min_score=0.0, score_fn=tie_score)

    assert await reranker.rerank("", chunks) == []
    assert await reranker.rerank("valid query", []) == []

    tied = await reranker.rerank("valid query", chunks)
    assert [c.chunk_id for c in tied] == ["c_high_rrf", "c_low_rrf"]