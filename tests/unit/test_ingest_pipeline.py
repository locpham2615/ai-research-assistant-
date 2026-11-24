from pathlib import Path
from typing import Any

from ai_research_assistant.db.sqlite import SQLiteStore
from ai_research_assistant.db.vector_store import VectorStore
from ai_research_assistant.pipelines.chunkers.section_chunker import SectionChunker
from ai_research_assistant.pipelines.embedders.text_embedder import TextEmbedder
from ai_research_assistant.pipelines.ingest_pipeline import IngestPipeline
from ai_research_assistant.pipelines.parsers.document_parser import DocumentParser
from ai_research_assistant.pipelines.schemas import PaperCard


class FakeSchemaGuard:
    def __init__(self) -> None:
        self.calls = 0

    async def complete_structured(self, **kwargs: Any) -> PaperCard:
        self.calls += 1
        return PaperCard(
            problem_statement="High inference latency.",
            methodology="Prefix caching.",
            key_contributions=["Faster TTFT"],
            datasets_or_benchmarks=["ShareGPT"],
            limitations=["Memory overhead"],
        )


def _fake_embed_3d(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        if "version two" in text.lower():
            vectors.append([0.0, 1.0, 0.0])
        else:
            vectors.append([1.0, 0.0, 0.0])
    return vectors


async def test_ingest_pipeline_idempotency_and_delete_then_upsert(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "test.db")
    await store.init_db()

    vs = VectorStore(collection_name="test_papers", in_memory=True)
    vs.ensure_collection(vector_size=3)

    guard = FakeSchemaGuard()
    embedder = TextEmbedder(sqlite_store=store, embed_fn=_fake_embed_3d)
    pipeline = IngestPipeline(
        sqlite_store=store,
        vector_store=vs,
        parser=DocumentParser(),
        chunker=SectionChunker(max_chars=500, overlap_chars=50),
        embedder=embedder,
        schema_guard=guard,  # type: ignore[arg-type]
    )

    doc_file = tmp_path / "sample.md"
    doc_file.write_text("Version one discusses KV cache compression.", encoding="utf-8")

    # 1. Nạp mới lần đầu -> indexed
    res1 = await pipeline.ingest_file(doc_file, topic_id="llm_sys", doc_id="doc_kv")
    assert res1.status == "indexed"
    assert res1.chunks_count == 1
    assert guard.calls == 1

    saved = await store.get_document("doc_kv")
    assert saved is not None and saved["status"] == "indexed"
    assert "Prefix caching." in saved["paper_card_json"]

    # 2. Nạp lại cùng nội dung -> skipped (không gọi LLM hay ghi Qdrant lại)
    res2 = await pipeline.ingest_file(doc_file, topic_id="llm_sys", doc_id="doc_kv")
    assert res2.status == "skipped"
    assert res2.skipped_reason == "already_indexed"
    assert guard.calls == 1

    # 3. Cập nhật nội dung file -> Delete-then-Upsert (xóa chunk cũ, ghi chunk mới)
    doc_file.write_text("Version two introduces paged attention.", encoding="utf-8")
    res3 = await pipeline.ingest_file(doc_file, topic_id="llm_sys", doc_id="doc_kv")
    assert res3.status == "indexed"
    assert guard.calls == 2

    # Tìm bằng vector của version one không còn thấy text cũ nữa
    hits = vs.search([0.0, 1.0, 0.0], limit=5, topic_id="llm_sys")
    assert len(hits) == 1
    assert "Version two" in hits[0]["payload"]["text"]