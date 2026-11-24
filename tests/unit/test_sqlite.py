import asyncio
from pathlib import Path

from ai_research_assistant.db.sqlite import SQLiteStore


async def test_embedding_cache_roundtrip(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "test.db")
    await store.init_db()

    assert await store.get_cached_embedding("missing_hash") is None

    vector = [0.1, 0.2, 0.3]
    await store.set_cached_embedding("hash_1", vector)
    assert await store.get_cached_embedding("hash_1") == vector


async def test_api_cache_ttl_expiration(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "test.db")
    await store.init_db()

    payload = {"papers": ["2401.00001"]}
    await store.set_api_cache("query_1", payload, ttl_seconds=1)
    assert await store.get_api_cache("query_1") == payload

    await asyncio.sleep(1.1)
    assert await store.get_api_cache("query_1") is None

async def test_topic_and_document_lifecycle(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "test.db")
    await store.init_db()

    topic = await store.create_or_get_topic("topic_1", "LLM Systems", "Inference papers")
    assert topic["id"] == "topic_1"
    assert topic["name"] == "LLM Systems"

    await store.upsert_document(
        doc_id="doc_1",
        topic_id="topic_1",
        title="Prefix Caching Paper",
        source_type="arxiv",
        source_uri="https://arxiv.org/abs/2401.00001",
        file_hash="sha256_abc",
        status="pending",
    )

    doc = await store.get_document_by_hash("topic_1", "sha256_abc")
    assert doc is not None
    assert doc["id"] == "doc_1"
    assert doc["status"] == "pending"

    await store.save_parsed_and_card(
        doc_id="doc_1",
        parsed_markdown="# Abstract\nTest content",
        paper_card_json='{"problem_statement": "latency"}',
        status="parsed",
    )
    await store.update_document_status("doc_1", "indexed")

    updated = await store.get_document("doc_1")
    assert updated is not None
    assert updated["status"] == "indexed"
    assert updated["parsed_markdown"].startswith("# Abstract")

    docs = await store.list_documents_by_topic("topic_1")
    assert len(docs) == 1