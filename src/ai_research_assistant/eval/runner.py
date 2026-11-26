import asyncio
import logging
import time
from collections.abc import Callable
from typing import Literal, Protocol

from ai_research_assistant.db.sqlite import SQLiteStore
from ai_research_assistant.db.vector_store import VectorStore
from ai_research_assistant.eval.judge import RAGJudge
from ai_research_assistant.eval.metrics import (
    aggregate_eval_results,
    compute_doc_recall_at_k,
    compute_evidence_hit_at_k,
    compute_mrr_at_k,
    compute_ndcg_at_k,
)
from ai_research_assistant.eval.schemas import (
    EvalDataset,
    EvalReport,
    EvalSample,
    SampleEvalResult,
    StageLatency,
)
from ai_research_assistant.harness.schema_guard import SchemaGuard
from ai_research_assistant.pipelines.embedders.text_embedder import TextEmbedder
from ai_research_assistant.rag.rag_engine import CitationGroundedRAG
from ai_research_assistant.retrieval.hybrid_retriever import HybridRetriever
from ai_research_assistant.retrieval.query_rewriter import QueryRewriter
from ai_research_assistant.retrieval.reranker import CrossEncoderReranker
from ai_research_assistant.retrieval.schemas import (
    RetrievalTrace,
    RetrievedChunk,
    RewrittenQuery,
)

logger = logging.getLogger(__name__)

EvalMode = Literal["dense_only", "hybrid_rrf", "full_rag"]
ProgressCallback = Callable[[int, int, SampleEvalResult], None]


class RetrieverProtocol(Protocol):
    async def retrieve(
        self,
        query: str,
        topic_id: str | None = None,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalTrace]: ...


class RerankerProtocol(Protocol):
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]: ...


class DenseOnlyRetriever:
    """Baseline thuần vector: tắt Query Rewrite, tắt BM25, tắt RRF."""

    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: VectorStore,
        retrieval_top_k: int = 20,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.retrieval_top_k = retrieval_top_k

    async def retrieve(
        self,
        query: str,
        topic_id: str | None = None,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        limit = top_k if top_k is not None else self.retrieval_top_k
        cleaned = query.strip()
        rewritten = RewrittenQuery(
            original_query=query,
            search_query=cleaned,
            bm25_keywords=[],
            used_fallback=False,
        )
        if not cleaned or limit <= 0:
            trace = RetrievalTrace(
                rewritten_query=rewritten,
                dense_candidates=0,
                bm25_candidates=0,
                fused_candidates=0,
            )
            return [], trace

        query_vec = await self.embedder.embed_query(cleaned)
        hits = await asyncio.to_thread(
            self.vector_store.search,
            query_vec,
            limit,
            topic_id,
            doc_ids,
        )

        chunks: list[RetrievedChunk] = []
        for hit in hits:
            payload = hit.get("payload") or {}
            doc_id = str(payload.get("doc_id", ""))
            chunk_index = int(payload.get("chunk_index", 0))
            chunk_id = str(hit.get("id") or f"{doc_id}:{chunk_index}")
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    topic_id=str(payload.get("topic_id", "")),
                    title=str(payload.get("title", "")),
                    section=str(payload.get("section", "")),
                    chunk_index=chunk_index,
                    text=str(payload.get("text", "")),
                    dense_score=float(hit.get("score", 0.0)),
                    rrf_score=0.0,
                )
            )

        trace = RetrievalTrace(
            rewritten_query=rewritten,
            dense_candidates=len(chunks),
            bm25_candidates=0,
            fused_candidates=len(chunks),
        )
        return chunks, trace


class PassThroughReranker:
    """Reranker giữ nguyên thứ tự và chỉ cắt top_k (dùng cho dense_only / hybrid_rrf)."""

    def __init__(self, top_k: int = 5) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.top_k = top_k

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        limit = top_k if top_k is not None else self.top_k
        if not query.strip() or not chunks or limit <= 0:
            return []
        return list(chunks[:limit])


class _InstrumentedRetriever:
    """Proxy đo retrieve_ms và lưu lại danh sách chunk trước bước Rerank."""

    def __init__(self, inner: RetrieverProtocol) -> None:
        self._inner = inner
        self.last_chunks: list[RetrievedChunk] = []
        self.last_ms: float = 0.0

    async def retrieve(
        self,
        query: str,
        topic_id: str | None = None,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        start = time.perf_counter()
        chunks, trace = await self._inner.retrieve(
            query=query,
            topic_id=topic_id,
            doc_ids=doc_ids,
            top_k=top_k,
        )
        self.last_ms = (time.perf_counter() - start) * 1000.0
        self.last_chunks = list(chunks)
        return chunks, trace


class _InstrumentedReranker:
    """Proxy đo rerank_ms và lưu lại danh sách chunk sau bước Rerank."""

    def __init__(self, inner: RerankerProtocol) -> None:
        self._inner = inner
        self.last_chunks: list[RetrievedChunk] = []
        self.last_ms: float = 0.0

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        start = time.perf_counter()
        reranked = await self._inner.rerank(
            query=query,
            chunks=chunks,
            top_k=top_k,
            min_score=min_score,
        )
        self.last_ms = (time.perf_counter() - start) * 1000.0
        self.last_chunks = list(reranked)
        return reranked


class EvalRunner:
    def __init__(
        self,
        mode: EvalMode,
        retriever: RetrieverProtocol,
        reranker: RerankerProtocol,
        schema_guard: SchemaGuard,
        judge: RAGJudge | None = None,
        retrieval_top_k: int = 20,
        rerank_top_k: int = 5,
        max_context_chars: int = 7500,
    ) -> None:
        self.mode = mode
        self._retriever = _InstrumentedRetriever(retriever)
        self._reranker = _InstrumentedReranker(reranker)
        self.rag = CitationGroundedRAG(
            retriever=self._retriever,  # type: ignore[arg-type]
            reranker=self._reranker,  # type: ignore[arg-type]
            schema_guard=schema_guard,
            max_context_chars=max_context_chars,
        )
        self.judge = judge
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k

    @classmethod
    def from_components(
        cls,
        mode: EvalMode,
        embedder: TextEmbedder,
        vector_store: VectorStore,
        sqlite_store: SQLiteStore,
        schema_guard: SchemaGuard,
        reranker_model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        retrieval_top_k: int = 20,
        rerank_top_k: int = 5,
        rerank_min_score: float = 0.0,
        rrf_k: int = 60,
        max_context_chars: int = 7500,
        judge: RAGJudge | None = None,
    ) -> "EvalRunner":
        """Khởi tạo EvalRunner tương ứng với từng cấu hình Ablation."""
        retriever: RetrieverProtocol
        reranker: RerankerProtocol

        if mode == "dense_only":
            retriever = DenseOnlyRetriever(
                embedder=embedder,
                vector_store=vector_store,
                retrieval_top_k=retrieval_top_k,
            )
            reranker = PassThroughReranker(top_k=rerank_top_k)
        elif mode == "hybrid_rrf":
            retriever = HybridRetriever(
                embedder=embedder,
                vector_store=vector_store,
                sqlite_store=sqlite_store,
                query_rewriter=QueryRewriter(schema_guard=schema_guard),
                retrieval_top_k=retrieval_top_k,
                rrf_k=rrf_k,
            )
            reranker = PassThroughReranker(top_k=rerank_top_k)
        else:
            retriever = HybridRetriever(
                embedder=embedder,
                vector_store=vector_store,
                sqlite_store=sqlite_store,
                query_rewriter=QueryRewriter(schema_guard=schema_guard),
                retrieval_top_k=retrieval_top_k,
                rrf_k=rrf_k,
            )
            reranker = CrossEncoderReranker(
                model_name=reranker_model_name,
                top_k=rerank_top_k,
                min_score=rerank_min_score,
            )

        return cls(
            mode=mode,
            retriever=retriever,
            reranker=reranker,
            schema_guard=schema_guard,
            judge=judge,
            retrieval_top_k=retrieval_top_k,
            rerank_top_k=rerank_top_k,
            max_context_chars=max_context_chars,
        )

    async def evaluate_sample(self, sample: EvalSample) -> SampleEvalResult:
        self._retriever.last_chunks = []
        self._retriever.last_ms = 0.0
        self._reranker.last_chunks = []
        self._reranker.last_ms = 0.0

        total_start = time.perf_counter()
        try:
            grounded = await self.rag.answer(
                query=sample.question,
                topic_id=sample.topic_id,
                top_k=self.rerank_top_k,
            )
        except Exception as exc:
            total_ms = (time.perf_counter() - total_start) * 1000.0
            logger.warning(
                "EvalRunner failed on %s (%s): %s",
                sample.sample_id,
                type(exc).__name__,
                exc,
            )
            return SampleEvalResult(
                sample_id=sample.sample_id,
                category=sample.category,
                should_abstain=sample.should_abstain,
                is_grounded=False,
                abstain_reason=f"runtime_error:{type(exc).__name__}",
                latency=StageLatency(
                    retrieve_ms=self._retriever.last_ms,
                    rerank_ms=self._reranker.last_ms,
                    generate_ms=max(
                        0.0, total_ms - self._retriever.last_ms - self._reranker.last_ms
                    ),
                    total_ms=total_ms,
                ),
            )

        total_ms = (time.perf_counter() - total_start) * 1000.0
        retrieve_ms = self._retriever.last_ms
        rerank_ms = self._reranker.last_ms
        generate_ms = (
            max(0.0, total_ms - retrieve_ms - rerank_ms)
            if self._reranker.last_chunks
            else 0.0
        )

        doc_recall_pre: float | None = None
        doc_recall_post: float | None = None
        evidence_pre: float | None = None
        evidence_post: float | None = None
        mrr_post: float | None = None
        ndcg_post: float | None = None

        if not sample.should_abstain:
            pre_chunks = self._retriever.last_chunks
            post_chunks = self._reranker.last_chunks
            doc_recall_pre = compute_doc_recall_at_k(
                pre_chunks,
                sample.expected_doc_ids,
                k=self.retrieval_top_k,
            )
            doc_recall_post = compute_doc_recall_at_k(
                post_chunks,
                sample.expected_doc_ids,
                k=self.rerank_top_k,
            )
            evidence_pre = compute_evidence_hit_at_k(
                pre_chunks,
                sample.expected_doc_ids,
                sample.expected_evidence_keywords,
                k=self.retrieval_top_k,
            )
            evidence_post = compute_evidence_hit_at_k(
                post_chunks,
                sample.expected_doc_ids,
                sample.expected_evidence_keywords,
                k=self.rerank_top_k,
            )
            mrr_post = compute_mrr_at_k(
                post_chunks,
                sample.expected_doc_ids,
                sample.expected_evidence_keywords,
                k=self.rerank_top_k,
            )
            ndcg_post = compute_ndcg_at_k(
                post_chunks,
                sample.expected_doc_ids,
                sample.expected_evidence_keywords,
                k=self.rerank_top_k,
            )

        judge_scores = (
            await self.judge.evaluate(sample, grounded)
            if self.judge is not None
            else None
        )

        return SampleEvalResult(
            sample_id=sample.sample_id,
            category=sample.category,
            should_abstain=sample.should_abstain,
            is_grounded=grounded.is_grounded,
            abstain_reason=grounded.abstain_reason,
            answer=grounded.answer,
            doc_recall_pre_rerank=doc_recall_pre,
            doc_recall_post_rerank=doc_recall_post,
            evidence_hit_pre_rerank=evidence_pre,
            evidence_hit_post_rerank=evidence_post,
            mrr_post_rerank=mrr_post,
            ndcg_post_rerank=ndcg_post,
            has_valid_citation=len(grounded.citations) > 0,
            invalid_citations_count=len(grounded.invalid_citations_removed),
            latency=StageLatency(
                retrieve_ms=retrieve_ms,
                rerank_ms=rerank_ms,
                generate_ms=generate_ms,
                total_ms=total_ms,
            ),
            judge=judge_scores,
        )

    async def run(
        self,
        dataset: EvalDataset,
        categories: set[str] | None = None,
        limit: int | None = None,
        on_sample_complete: ProgressCallback | None = None,
    ) -> EvalReport:
        samples = [
            s
            for s in dataset.samples
            if categories is None or s.category in categories
        ]
        if limit is not None and limit > 0:
            samples = samples[:limit]

        results: list[SampleEvalResult] = []
        total = len(samples)
        for idx, sample in enumerate(samples, start=1):
            result = await self.evaluate_sample(sample)
            results.append(result)
            if on_sample_complete is not None:
                on_sample_complete(idx, total, result)

        return aggregate_eval_results(
            dataset_name=dataset.dataset_name,
            mode=self.mode,
            results=results,
        )