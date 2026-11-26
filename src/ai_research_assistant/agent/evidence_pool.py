from typing import Protocol

from pydantic import BaseModel, Field

from ai_research_assistant.agent.schemas import SubQuery
from ai_research_assistant.retrieval.schemas import (
    RetrievalTrace,
    RetrievedChunk,
    RewrittenQuery,
)


class SubQueryRetrieverProtocol(Protocol):
    async def retrieve(
        self,
        query: str,
        topic_id: str | None = None,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
        rewritten_query: RewrittenQuery | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalTrace]: ...


class RerankerProtocol(Protocol):
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]: ...


class BatchRetrievalResult(BaseModel):
    pre_rerank_chunks: list[RetrievedChunk] = Field(default_factory=list)
    chunks_by_sub_query: dict[str, list[RetrievedChunk]] = Field(default_factory=dict)
    dense_candidates_total: int = 0
    bm25_candidates_total: int = 0


def _chunk_sort_key(chunk: RetrievedChunk) -> tuple[float, float]:
    rerank = chunk.rerank_score if chunk.rerank_score is not None else float("-inf")
    return (rerank, chunk.rrf_score)


def merge_pre_rerank_chunks(
    existing: list[RetrievedChunk],
    incoming: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Gộp danh sách ứng viên trước rerank, khử trùng lặp và giữ điểm tốt hơn."""
    by_id: dict[str, RetrievedChunk] = {c.chunk_id: c for c in existing}
    for chunk in incoming:
        prev = by_id.get(chunk.chunk_id)
        if prev is None or chunk.rrf_score > prev.rrf_score:
            by_id[chunk.chunk_id] = chunk

    merged = list(by_id.values())
    merged.sort(key=lambda c: (c.rrf_score, c.dense_score or 0.0), reverse=True)
    return merged


def merge_evidence_interleaved(
    chunks_by_sub_query: dict[str, list[RetrievedChunk]],
    top_k: int = 6,
) -> list[RetrievedChunk]:
    """
    Trộn xen kẽ (cursor-based round-robin) giữa các sub-query sau rerank:
    - Bảo vệ suất cân bằng cho từng bài báo/khía cạnh trong câu hỏi multi_paper.
    - Khử trùng lặp theo chunk_id và giữ điểm rerank cao nhất từng quan sát.
    """
    if top_k <= 0 or not chunks_by_sub_query:
        return []

    best_by_id: dict[str, RetrievedChunk] = {}
    ordered_lists: list[list[RetrievedChunk]] = []

    for chunks in chunks_by_sub_query.values():
        sorted_chunks = sorted(chunks, key=_chunk_sort_key, reverse=True)
        ordered_lists.append(sorted_chunks)
        for chunk in sorted_chunks:
            prev = best_by_id.get(chunk.chunk_id)
            if prev is None or _chunk_sort_key(chunk) > _chunk_sort_key(prev):
                best_by_id[chunk.chunk_id] = chunk

    pointers = [0] * len(ordered_lists)
    seen_ids: set[str] = set()
    merged: list[RetrievedChunk] = []

    while len(merged) < top_k:
        progressed = False
        for idx, sub_list in enumerate(ordered_lists):
            ptr = pointers[idx]
            while ptr < len(sub_list) and sub_list[ptr].chunk_id in seen_ids:
                ptr += 1

            if ptr < len(sub_list):
                chosen_id = sub_list[ptr].chunk_id
                pointers[idx] = ptr + 1
                seen_ids.add(chosen_id)
                merged.append(best_by_id[chosen_id])
                progressed = True
                if len(merged) >= top_k:
                    break
            else:
                pointers[idx] = ptr

        if not progressed:
            break

    return merged


class EvidencePool:
    def __init__(
        self,
        retriever: SubQueryRetrieverProtocol,
        reranker: RerankerProtocol,
        per_sub_query_top_k: int = 4,
        merged_top_k: int = 6,
    ) -> None:
        if per_sub_query_top_k <= 0:
            raise ValueError("per_sub_query_top_k must be positive")
        if merged_top_k <= 0:
            raise ValueError("merged_top_k must be positive")
        self.retriever = retriever
        self.reranker = reranker
        self.per_sub_query_top_k = per_sub_query_top_k
        self.merged_top_k = merged_top_k

    async def retrieve_batch(
        self,
        sub_queries: list[SubQuery],
        topic_id: str | None = None,
        doc_ids: list[str] | None = None,
        per_query_top_k: int | None = None,
    ) -> BatchRetrievalResult:
        if not sub_queries:
            return BatchRetrievalResult()

        # Nếu chỉ có 1 truy vấn (nhánh simple), lấy đủ chiều sâu tới merged_top_k
        default_limit = (
            self.merged_top_k if len(sub_queries) == 1 else self.per_sub_query_top_k
        )
        rerank_limit = per_query_top_k if per_query_top_k is not None else default_limit

        pre_rerank: list[RetrievedChunk] = []
        by_sub_query: dict[str, list[RetrievedChunk]] = {}
        dense_total = 0
        bm25_total = 0

        for sq in sub_queries:
            rewritten = RewrittenQuery(
                original_query=sq.search_query,
                search_query=sq.search_query,
                bm25_keywords=sq.bm25_keywords,
                used_fallback=False,
            )
            candidates, trace = await self.retriever.retrieve(
                query=sq.search_query,
                topic_id=topic_id,
                doc_ids=doc_ids,
                rewritten_query=rewritten,
            )
            reranked = await self.reranker.rerank(
                query=sq.search_query,
                chunks=candidates,
                top_k=rerank_limit,
            )

            dense_total += trace.dense_candidates
            bm25_total += trace.bm25_candidates
            pre_rerank = merge_pre_rerank_chunks(pre_rerank, candidates)
            by_sub_query[sq.sub_query_id] = reranked

        return BatchRetrievalResult(
            pre_rerank_chunks=pre_rerank,
            chunks_by_sub_query=by_sub_query,
            dense_candidates_total=dense_total,
            bm25_candidates_total=bm25_total,
        )