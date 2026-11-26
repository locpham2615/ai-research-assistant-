import math
import re
from collections import defaultdict
from datetime import UTC, datetime

from ai_research_assistant.eval.dataset import normalize_doc_id
from ai_research_assistant.eval.schemas import (
    AbstentionMetrics,
    CategorySummary,
    CitationMetrics,
    EvalReport,
    JudgeSummary,
    LatencyPercentile,
    LatencySummary,
    RetrievalMetrics,
    SampleEvalResult,
)
from ai_research_assistant.retrieval.schemas import RetrievedChunk

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def _normalize_keywords(keywords: list[str]) -> list[str]:
    return [_normalize_text(kw) for kw in keywords if kw.strip()]


def _normalize_doc_set(doc_ids: list[str]) -> set[str]:
    return {normalize_doc_id(doc_id) for doc_id in doc_ids if doc_id.strip()}


def _is_relevant_chunk(
    chunk: RetrievedChunk,
    expected_docs: set[str],
    normalized_keywords: list[str],
) -> bool:
    if normalize_doc_id(chunk.doc_id) not in expected_docs:
        return False
    if not normalized_keywords:
        return True
    haystack = _normalize_text(f"{chunk.section} {chunk.text}")
    return any(kw in haystack for kw in normalized_keywords)


def compute_doc_recall_at_k(
    chunks: list[RetrievedChunk],
    expected_doc_ids: list[str],
    k: int,
) -> float:
    expected = _normalize_doc_set(expected_doc_ids)
    if not expected or k <= 0:
        return 0.0
    retrieved_docs = {normalize_doc_id(c.doc_id) for c in chunks[:k]}
    return len(expected & retrieved_docs) / len(expected)


def compute_evidence_hit_at_k(
    chunks: list[RetrievedChunk],
    expected_doc_ids: list[str],
    expected_keywords: list[str],
    k: int,
) -> float:
    """Tỷ lệ từ khóa bằng chứng xuất hiện trong top-K chunk thuộc các bài báo mục tiêu."""
    expected_docs = _normalize_doc_set(expected_doc_ids)
    if not expected_docs or k <= 0:
        return 0.0

    matching_chunks = [
        c for c in chunks[:k] if normalize_doc_id(c.doc_id) in expected_docs
    ]
    if not matching_chunks:
        return 0.0

    keywords = _normalize_keywords(expected_keywords)
    if not keywords:
        return 1.0

    combined = _normalize_text(
        " ".join(f"{c.section} {c.text}" for c in matching_chunks)
    )
    hits = sum(1 for kw in keywords if kw in combined)
    return hits / len(keywords)


def compute_mrr_at_k(
    chunks: list[RetrievedChunk],
    expected_doc_ids: list[str],
    expected_keywords: list[str],
    k: int,
) -> float:
    expected_docs = _normalize_doc_set(expected_doc_ids)
    if not expected_docs or k <= 0:
        return 0.0
    keywords = _normalize_keywords(expected_keywords)

    for rank, chunk in enumerate(chunks[:k], start=1):
        if _is_relevant_chunk(chunk, expected_docs, keywords):
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(
    chunks: list[RetrievedChunk],
    expected_doc_ids: list[str],
    expected_keywords: list[str],
    k: int,
) -> float:
    expected_docs = _normalize_doc_set(expected_doc_ids)
    if not expected_docs or k <= 0:
        return 0.0
    keywords = _normalize_keywords(expected_keywords)

    rels = [
        1.0 if _is_relevant_chunk(chunk, expected_docs, keywords) else 0.0
        for chunk in chunks[:k]
    ]
    dcg = sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(rels, start=1)
        if rel > 0.0
    )
    if dcg == 0.0:
        return 0.0

    # Đảm bảo câu hỏi multi_paper không đạt 1.0 nếu mới tìm thấy bằng chứng từ 1 bài
    ideal_hits = min(k, max(len(expected_docs), int(sum(rels))))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0.0 else 0.0


def compute_abstention_metrics(results: list[SampleEvalResult]) -> AbstentionMetrics:
    tp = fp = tn = fn = 0
    for r in results:
        predicted_abstain = not r.is_grounded
        if r.should_abstain and predicted_abstain:
            tp += 1
        elif not r.should_abstain and predicted_abstain:
            fp += 1
        elif not r.should_abstain and not predicted_abstain:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )
    false_refusal_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    unanswerable_leak_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    return AbstentionMetrics(
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        false_refusal_rate=false_refusal_rate,
        unanswerable_leak_rate=unanswerable_leak_rate,
    )


def compute_citation_metrics(results: list[SampleEvalResult]) -> CitationMetrics:
    answerable = [r for r in results if not r.should_abstain]
    grounded_answerable = [r for r in answerable if r.is_grounded]

    grounded_rate = (
        len(grounded_answerable) / len(answerable) if answerable else 0.0
    )
    valid_rate = (
        sum(1 for r in grounded_answerable if r.has_valid_citation)
        / len(grounded_answerable)
        if grounded_answerable
        else 0.0
    )

    total_invalid = sum(r.invalid_citations_count for r in results)
    hallucinated_rate = (
        sum(1 for r in results if r.invalid_citations_count > 0) / len(results)
        if results
        else 0.0
    )

    return CitationMetrics(
        grounded_answer_rate=grounded_rate,
        valid_citation_rate=valid_rate,
        hallucinated_citation_rate=hallucinated_rate,
        total_invalid_citations=total_invalid,
    )


def compute_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * (percentile / 100.0)
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return float(ordered[low])
    return ordered[low] + (ordered[high] - ordered[low]) * (idx - low)


def _summarize_latency_series(values: list[float]) -> LatencyPercentile:
    if not values:
        return LatencyPercentile()
    return LatencyPercentile(
        mean_ms=sum(values) / len(values),
        p50_ms=compute_percentile(values, 50.0),
        p95_ms=compute_percentile(values, 95.0),
    )


def compute_latency_summary(results: list[SampleEvalResult]) -> LatencySummary:
    return LatencySummary(
        retrieve=_summarize_latency_series([r.latency.retrieve_ms for r in results]),
        rerank=_summarize_latency_series([r.latency.rerank_ms for r in results]),
        generate=_summarize_latency_series([r.latency.generate_ms for r in results]),
        total=_summarize_latency_series([r.latency.total_ms for r in results]),
    )


def _mean_optional(values: list[float | None]) -> float:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else 0.0


def aggregate_eval_results(
    dataset_name: str,
    mode: str,
    results: list[SampleEvalResult],
) -> EvalReport:
    answerable = [r for r in results if not r.should_abstain]
    unanswerable = [r for r in results if r.should_abstain]

    retrieval = RetrievalMetrics(
        doc_recall_pre_rerank=_mean_optional(
            [r.doc_recall_pre_rerank for r in answerable]
        ),
        doc_recall_post_rerank=_mean_optional(
            [r.doc_recall_post_rerank for r in answerable]
        ),
        evidence_hit_pre_rerank=_mean_optional(
            [r.evidence_hit_pre_rerank for r in answerable]
        ),
        evidence_hit_post_rerank=_mean_optional(
            [r.evidence_hit_post_rerank for r in answerable]
        ),
        mrr_post_rerank=_mean_optional([r.mrr_post_rerank for r in answerable]),
        ndcg_post_rerank=_mean_optional([r.ndcg_post_rerank for r in answerable]),
    )

    judged = [r.judge for r in results if r.judge is not None]
    judge_summary: JudgeSummary | None = None
    if judged:
        judge_summary = JudgeSummary(
            evaluated_count=len(judged),
            mean_faithfulness=sum(j.faithfulness for j in judged) / len(judged),
            mean_correctness=sum(j.correctness for j in judged) / len(judged),
            mean_relevancy=sum(j.relevancy for j in judged) / len(judged),
        )

    grouped: dict[str, list[SampleEvalResult]] = defaultdict(list)
    for r in results:
        grouped[r.category].append(r)

    by_category: dict[str, CategorySummary] = {}
    for category, items in sorted(grouped.items()):
        ans_items = [i for i in items if not i.should_abstain]
        by_category[category] = CategorySummary(
            count=len(items),
            grounded_rate=sum(1 for i in items if i.is_grounded) / len(items),
            doc_recall_post_rerank=(
                _mean_optional([i.doc_recall_post_rerank for i in ans_items])
                if ans_items
                else None
            ),
            evidence_hit_post_rerank=(
                _mean_optional([i.evidence_hit_post_rerank for i in ans_items])
                if ans_items
                else None
            ),
            mrr_post_rerank=(
                _mean_optional([i.mrr_post_rerank for i in ans_items])
                if ans_items
                else None
            ),
        )

    return EvalReport(
        dataset_name=dataset_name,
        mode=mode,
        created_at=datetime.now(UTC).isoformat(),
        total_samples=len(results),
        answerable_samples=len(answerable),
        unanswerable_samples=len(unanswerable),
        retrieval=retrieval,
        citation=compute_citation_metrics(results),
        abstention=compute_abstention_metrics(results),
        latency=compute_latency_summary(results),
        judge=judge_summary,
        by_category=by_category,
        results=results,
    )