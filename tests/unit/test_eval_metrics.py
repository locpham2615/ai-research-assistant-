import math

from ai_research_assistant.eval.metrics import (
    aggregate_eval_results,
    compute_abstention_metrics,
    compute_citation_metrics,
    compute_doc_recall_at_k,
    compute_evidence_hit_at_k,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_percentile,
)
from ai_research_assistant.eval.schemas import (
    JudgeScores,
    SampleEvalResult,
    StageLatency,
)
from ai_research_assistant.retrieval.schemas import RetrievedChunk


def _chunk(doc_id: str, idx: int, text: str, section: str = "Method") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}_{idx}",
        doc_id=doc_id,
        topic_id="llm_eval_v1",
        title="Paper",
        section=section,
        chunk_index=idx,
        text=text,
        rrf_score=0.03,
    )


def test_retrieval_metrics_version_normalization_and_pdf_whitespace() -> None:
    chunks = [
        _chunk("arxiv_2106.09685v2", 0, "LoRA uses low-rank matrices."),
        _chunk(
            "arxiv_1706.03762v7",
            1,
            "We scale the dot\nproducts by 1/sqrt(d_k) to prevent small softmax gradients.",
        ),
    ]
    expected_docs = ["arxiv_1706.03762"]
    keywords = ["dot products", "softmax", "gradients"]

    assert compute_doc_recall_at_k(chunks, expected_docs, k=1) == 0.0
    assert compute_doc_recall_at_k(chunks, expected_docs, k=2) == 1.0
    assert compute_evidence_hit_at_k(chunks, expected_docs, keywords, k=2) == 1.0
    assert compute_mrr_at_k(chunks, expected_docs, keywords, k=2) == 0.5

    expected_ndcg = (1.0 / math.log2(3)) / 1.0
    assert abs(compute_ndcg_at_k(chunks, expected_docs, keywords, k=2) - expected_ndcg) < 1e-9


def test_retrieval_metrics_multi_paper_combines_keywords_and_penalizes_partial_ndcg() -> None:
    chunks = [
        _chunk("arxiv_2106.09685v2", 0, "LoRA injects trainable rank decomposition matrices."),
        _chunk("arxiv_2305.14314v1", 1, "QLoRA introduces NF4 and double quantization."),
    ]
    expected_docs = ["arxiv_2106.09685", "arxiv_2305.14314"]
    keywords = ["rank decomposition", "nf4", "double quantization"]

    # Chỉ xét top 1 -> tìm được 1/2 bài, 1/3 từ khóa, NDCG < 1.0
    assert compute_doc_recall_at_k(chunks, expected_docs, k=1) == 0.5
    assert abs(compute_evidence_hit_at_k(chunks, expected_docs, keywords, k=1) - (1 / 3)) < 1e-9

    # Xét top 2 -> đủ 2/2 bài và 3/3 từ khóa, NDCG = 1.0
    assert compute_doc_recall_at_k(chunks, expected_docs, k=2) == 1.0
    assert compute_evidence_hit_at_k(chunks, expected_docs, keywords, k=2) == 1.0
    assert compute_ndcg_at_k(chunks, expected_docs, keywords, k=2) == 1.0

    # Nếu chỉ có 1 bài trong top 2 -> NDCG bị trừ điểm vì thiếu bài thứ 2
    partial = [chunks[0], _chunk("arxiv_2005.11401v4", 0, "Unrelated RAG text.")]
    expected_partial_ndcg = 1.0 / (1.0 + 1.0 / math.log2(3))
    assert (
        abs(compute_ndcg_at_k(partial, expected_docs, keywords, k=2) - expected_partial_ndcg)
        < 1e-9
    )


def test_abstention_citation_latency_and_aggregation() -> None:
    results = [
        # TN: answerable & grounded + valid citation
        SampleEvalResult(
            sample_id="q001",
            category="factoid",
            should_abstain=False,
            is_grounded=True,
            doc_recall_pre_rerank=1.0,
            doc_recall_post_rerank=1.0,
            evidence_hit_pre_rerank=1.0,
            evidence_hit_post_rerank=1.0,
            mrr_post_rerank=1.0,
            ndcg_post_rerank=1.0,
            has_valid_citation=True,
            invalid_citations_count=0,
            latency=StageLatency(retrieve_ms=100, rerank_ms=50, generate_ms=800, total_ms=950),
            judge=JudgeScores(faithfulness=1.0, correctness=0.9, relevancy=1.0),
        ),
        # FP (False Refusal): answerable nhưng hệ thống từ chối oan
        SampleEvalResult(
            sample_id="q002",
            category="conceptual",
            should_abstain=False,
            is_grounded=False,
            abstain_reason="no_relevant_evidence",
            doc_recall_pre_rerank=1.0,
            doc_recall_post_rerank=0.0,
            evidence_hit_pre_rerank=0.5,
            evidence_hit_post_rerank=0.0,
            mrr_post_rerank=0.0,
            ndcg_post_rerank=0.0,
            has_valid_citation=False,
            invalid_citations_count=0,
            latency=StageLatency(retrieve_ms=120, rerank_ms=60, generate_ms=0, total_ms=180),
        ),
        # TP: unanswerable và hệ thống từ chối đúng
        SampleEvalResult(
            sample_id="q003",
            category="unanswerable",
            should_abstain=True,
            is_grounded=False,
            abstain_reason="no_relevant_evidence",
            latency=StageLatency(retrieve_ms=110, rerank_ms=55, generate_ms=0, total_ms=165),
        ),
        # FN (Leak): unanswerable nhưng hệ thống vẫn trả lời + bịa citation
        SampleEvalResult(
            sample_id="q004",
            category="unanswerable",
            should_abstain=True,
            is_grounded=True,
            has_valid_citation=True,
            invalid_citations_count=2,
            latency=StageLatency(retrieve_ms=140, rerank_ms=70, generate_ms=900, total_ms=1110),
        ),
    ]

    ab = compute_abstention_metrics(results)
    assert (ab.true_positives, ab.false_positives, ab.true_negatives, ab.false_negatives) == (
        1,
        1,
        1,
        1,
    )
    assert ab.precision == 0.5
    assert ab.recall == 0.5
    assert ab.f1 == 0.5
    assert ab.false_refusal_rate == 0.5
    assert ab.unanswerable_leak_rate == 0.5

    cit = compute_citation_metrics(results)
    assert cit.grounded_answer_rate == 0.5
    assert cit.valid_citation_rate == 1.0
    assert cit.hallucinated_citation_rate == 0.25
    assert cit.total_invalid_citations == 2

    assert compute_percentile([], 50.0) == 0.0
    assert compute_percentile([10.0, 20.0, 30.0, 40.0], 50.0) == 25.0

    report = aggregate_eval_results("rag_golden_v1", "full_rag", results)
    assert report.total_samples == 4
    assert report.answerable_samples == 2
    assert report.unanswerable_samples == 2
    assert report.retrieval.doc_recall_pre_rerank == 1.0
    assert report.retrieval.doc_recall_post_rerank == 0.5
    assert report.judge is not None
    assert report.judge.evaluated_count == 1
    assert report.by_category["unanswerable"].doc_recall_post_rerank is None
    assert report.by_category["factoid"].mrr_post_rerank == 1.0