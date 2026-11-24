import json
import time
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS topics (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    topic_id TEXT NOT NULL,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL,          -- 'arxiv' | 'upload'
    source_uri TEXT DEFAULT '',
    file_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending', -- 'pending' | 'parsed' | 'indexed' | 'failed'
    parsed_markdown TEXT DEFAULT '',
    paper_card_json TEXT DEFAULT '',
    error_message TEXT DEFAULT '',
    updated_at REAL NOT NULL,
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

CREATE TABLE IF NOT EXISTS comparison_matrices (
    cache_key TEXT PRIMARY KEY,         -- sorted doc_ids hash
    topic_id TEXT NOT NULL,
    matrix_markdown TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS user_memory (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS embedding_cache (
    text_hash TEXT PRIMARY KEY,
    vector_json TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS api_cache (
    query_hash TEXT PRIMARY KEY,
    response_json TEXT NOT NULL,
    expires_at REAL NOT NULL
);
"""


class SQLiteStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = str(db_path)

    async def init_db(self) -> None:
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA_SQL)
            await db.commit()

    # --- Embedding Cache ---
    async def get_cached_embedding(self, text_hash: str) -> list[float] | None:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT vector_json FROM embedding_cache WHERE text_hash = ?",
                (text_hash,),
            )
            row = await cursor.fetchone()
            return json.loads(row[0]) if row else None

    async def set_cached_embedding(self, text_hash: str, vector: list[float]) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO embedding_cache (text_hash, vector_json, created_at)
                VALUES (?, ?, ?)
                """,
                (text_hash, json.dumps(vector), time.time()),
            )
            await db.commit()

    # --- External API TTL Cache (arXiv / Semantic Scholar) ---
    async def get_api_cache(self, query_hash: str) -> Any | None:
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT response_json, expires_at FROM api_cache WHERE query_hash = ?",
                (query_hash,),
            )
            row = await cursor.fetchone()
            if not row:
                return None
            response_json, expires_at = row
            if now > expires_at:
                await db.execute("DELETE FROM api_cache WHERE query_hash = ?", (query_hash,))
                await db.commit()
                return None
            return json.loads(response_json)

    async def set_api_cache(self, query_hash: str, payload: Any, ttl_seconds: int = 21600) -> None:
        expires_at = time.time() + ttl_seconds
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO api_cache (query_hash, response_json, expires_at)
                VALUES (?, ?, ?)
                """,
                (query_hash, json.dumps(payload), expires_at),
            )
            await db.commit()