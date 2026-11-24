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