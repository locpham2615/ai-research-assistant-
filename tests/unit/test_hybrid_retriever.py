from pathlib import Path
from typing import Any

from qdrant_client.models import PointStruct

from ai_research_assistant.db.sqlite import SQLiteStore
from ai_research_assistant.db.vector_store import VectorStore
from ai_research_assistant.pipelines.embedders.text_embedder import TextEmbedder
from ai_research_assistant.retrieval.hybrid_retriever import HybridRetriever, fuse_rrf
from ai_research_assistant.retrieval.query_rewriter import QueryRewriter
from ai_research_assistant.retrieval.schemas import QueryRewriteOutput


class FakeRewriteGuard:
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail

    async def complete_structured(self, **kwargs: Any) -> QueryRewriteOutput:
        if self.should_fail:
            raise RuntimeError("LLM timeout")
        return QueryRewriteOutput(
            search_query="Why does scaled dot-product attention divide by sqrt d_k?",
            bm25_keywords=["scaled dot-product", "sqrt", "softmax", "sqrt"],
        )


async def test_query_rewriter_success_and_fallback() -> None:
    rewriter = QueryRewriter(schema_guard=FakeRewriteGuard())  # type: ignore[arg-type]
    ok = await rewriter.rewrite("Tại sao attention lại chia cho căn d_k?")
    assert ok.used_fallback is False
    assert "scaled dot-product" in ok.search_query
    assert ok.bm25_keywords == ["scaled dot-product", "sqrt", "softmax"]

    failing = QueryRewriter(
        schema_guard=FakeRewriteGuard(should_fail=True)  # type: ignore[arg-type]
    )
    fb = await failing.rewrite("Original query text")
    assert fb.used_fallback is True
    assert fb.search_query == "Original query text"
    assert fb.bm25_keywords == []


def test_fuse_rrf_boosts_overlap_and_deduplicates() -> None:
    dense_hits = [
        {
            "id": "c_only_dense",
            "score": 0.92,
            "payload": {
                "doc_id": "d1",
                "topic_id": "t1",
                "title": "Paper",
                "section": "Intro",
                "chunk_index": 0,
                "text": "Dense top 1",
            },
        },
        {
            "id": "c_both",
            "score": 0.85,
            "payload": {
                "doc_id": "d1",
                "topic_id": "t1",
                "title": "Paper",
                "section": "Method",
                "chunk_index": 1,
                "text": "Both dense and bm25",
            },
        },
    ]
    bm25_hits = [
        {
            "chunk_id": "c_both",
            "doc_id": "d1",
            "topic_id": "t1",
            "title": "Paper",
            "section": "Method",
            "chunk_index": 1,
            "text": "Both dense and bm25",
            "lexical_score": 4.2,
        },
        {
            "chunk_id": "c_only_bm25",
            "doc_id": "d1",
            "topic_id": "t1",
            "title": "Paper",
            "section": "Exp",
            "chunk_index": 2,
            "text": "BM25 only",
            "lexical_score": 2.1,
        },
    ]

    fused = fuse_rrf(dense_hits, bm25_hits, rrf_k=60, limit=10)
    assert [c.chunk_id for c in fused] == ["c_both", "c_only_dense", "c_only_bm25"]
    assert fused[0].dense_score == 0.85
    assert fused[0].lexical_score == 4.2
    expected_rrf = 1.0 / 62 + 1.0 / 61
    assert abs(fused[0].rrf_score - expected_rrf) < 1e-9


async def test_hybrid_retriever_end_to_end_offline(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "test.db")
    await store.init_db()

    vs = VectorStore(collection_name="test_hybrid", in_memory=True)
    vs.ensure_collection(vector_size=3)

    chunks = [
        {
            "chunk_id": "00000000-0000-0000-0000-000000000001",
            "doc_id": "doc_attn",
            "topic_id": "transformers",
            "title": "Attention Is All You Need",
            "section": "Scaled Dot-Product Attention",
            "chunk_index": 0,
            "text": "We divide dot products by sqrt d_k to prevent tiny softmax gradients.",
            "text_hash": "h1",
            "vector": [1.0, 0.0, 0.0],
        },
        {
            "chunk_id": "00000000-0000-0000-0000-000000000002",
            "doc_id": "doc_opt",
            "topic_id": "transformers",
            "title": "Attention Is All You Need",
            "section": "Optimizer",
            "chunk_index": 1,
            "text": "We train using Adam optimizer with warmup learning rate.",
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

    def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    embedder = TextEmbedder(sqlite_store=store, embed_fn=fake_embed)
    rewriter = QueryRewriter(schema_guard=FakeRewriteGuard())  # type: ignore[arg-type]
    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vs,
        sqlite_store=store,
        query_rewriter=rewriter,
        retrieval_top_k=5,
    )

    results, trace = await retriever.retrieve(
        "Tại sao phải chia cho căn d_k?",
        topic_id="transformers",
    )
    assert len(results) >= 1
    assert results[0].chunk_id == "00000000-0000-0000-0000-000000000001"
    assert results[0].dense_score is not None
    assert results[0].lexical_score is not None
    assert trace.dense_candidates == 2
    assert trace.bm25_candidates == 1
    assert trace.rewritten_query.used_fallback is False

    # Kiểm tra lọc theo doc_ids
    filtered, _ = await retriever.retrieve(
        "optimizer",
        topic_id="transformers",
        doc_ids=["doc_opt"],
    )
    assert all(r.doc_id == "doc_opt" for r in filtered)