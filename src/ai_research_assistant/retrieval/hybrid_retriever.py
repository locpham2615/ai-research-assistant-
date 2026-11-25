import asyncio
import uuid
from typing import Any

from ai_research_assistant.db.sqlite import SQLiteStore
from ai_research_assistant.db.vector_store import VectorStore
from ai_research_assistant.pipelines.embedders.text_embedder import TextEmbedder
from ai_research_assistant.retrieval.query_rewriter import QueryRewriter
from ai_research_assistant.retrieval.schemas import (
    RetrievalTrace,
    RetrievedChunk,
    RewrittenQuery,
)


def _fallback_chunk_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{chunk_index}"))


def fuse_rrf(
    dense_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    rrf_k: int = 60,
    limit: int = 20,
) -> list[RetrievedChunk]:
    """Hợp nhất kết quả Dense (Qdrant) và BM25 (SQLite FTS5) bằng RRF."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if limit <= 0:
        return []

    merged: dict[str, RetrievedChunk] = {}

    # 1. Nạp kết quả Dense (rank bắt đầu từ 1)
    for rank, hit in enumerate(dense_hits, start=1):
        payload = hit.get("payload") or {}
        doc_id = str(payload.get("doc_id", ""))
        chunk_index = int(payload.get("chunk_index", 0))
        chunk_id = str(hit.get("id") or _fallback_chunk_id(doc_id, chunk_index))
        dense_score = float(hit.get("score", 0.0))
        rrf_inc = 1.0 / (rrf_k + rank)

        if chunk_id in merged:
            merged[chunk_id].dense_score = dense_score
            merged[chunk_id].rrf_score += rrf_inc
        else:
            merged[chunk_id] = RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                topic_id=str(payload.get("topic_id", "")),
                title=str(payload.get("title", "")),
                section=str(payload.get("section", "")),
                chunk_index=chunk_index,
                text=str(payload.get("text", "")),
                dense_score=dense_score,
                rrf_score=rrf_inc,
            )

    # 2. Nạp kết quả BM25 (rank bắt đầu từ 1)
    for rank, hit in enumerate(bm25_hits, start=1):
        doc_id = str(hit.get("doc_id", ""))
        chunk_index = int(hit.get("chunk_index", 0))
        chunk_id = str(hit.get("chunk_id") or _fallback_chunk_id(doc_id, chunk_index))
        lexical_score = float(hit.get("lexical_score", 0.0))
        rrf_inc = 1.0 / (rrf_k + rank)

        if chunk_id in merged:
            merged[chunk_id].lexical_score = lexical_score
            merged[chunk_id].rrf_score += rrf_inc
        else:
            merged[chunk_id] = RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                topic_id=str(hit.get("topic_id", "")),
                title=str(hit.get("title", "")),
                section=str(hit.get("section", "")),
                chunk_index=chunk_index,
                text=str(hit.get("text", "")),
                lexical_score=lexical_score,
                rrf_score=rrf_inc,
            )

    ranked = sorted(
        merged.values(),
        key=lambda c: (
            c.rrf_score,
            c.dense_score if c.dense_score is not None else float("-inf"),
            c.lexical_score if c.lexical_score is not None else float("-inf"),
        ),
        reverse=True,
    )
    return ranked[:limit]


class HybridRetriever:
    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: VectorStore,
        sqlite_store: SQLiteStore,
        query_rewriter: QueryRewriter | None = None,
        retrieval_top_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.sqlite_store = sqlite_store
        self.query_rewriter = query_rewriter
        self.retrieval_top_k = retrieval_top_k
        self.rrf_k = rrf_k

    async def _search_dense(
        self,
        search_query: str,
        topic_id: str | None,
        doc_ids: list[str] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not search_query.strip():
            return []
        query_vec = await self.embedder.embed_query(search_query)
        return await asyncio.to_thread(
            self.vector_store.search,
            query_vec,
            limit,
            topic_id,
            doc_ids,
        )

    async def retrieve(
        self,
        query: str,
        topic_id: str | None = None,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        limit = top_k if top_k is not None else self.retrieval_top_k

        if self.query_rewriter is not None:
            rewritten = await self.query_rewriter.rewrite(query)
        else:
            rewritten = RewrittenQuery(
                original_query=query,
                search_query=query.strip(),
                bm25_keywords=[],
                used_fallback=False,
            )

        bm25_query = (
            " ".join(rewritten.bm25_keywords)
            if rewritten.bm25_keywords
            else rewritten.search_query
        )

        dense_hits, bm25_hits = await asyncio.gather(
            self._search_dense(rewritten.search_query, topic_id, doc_ids, limit),
            self.sqlite_store.search_bm25(
                bm25_query,
                topic_id=topic_id,
                doc_ids=doc_ids,
                limit=limit,
            ),
        )

        fused = fuse_rrf(
            dense_hits=dense_hits,
            bm25_hits=bm25_hits,
            rrf_k=self.rrf_k,
            limit=limit,
        )
        trace = RetrievalTrace(
            rewritten_query=rewritten,
            dense_candidates=len(dense_hits),
            bm25_candidates=len(bm25_hits),
            fused_candidates=len(fused),
        )
        return fused, trace