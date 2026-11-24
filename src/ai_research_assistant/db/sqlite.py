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

    # --- Topics CRUD ---
    async def create_or_get_topic(
        self,
        topic_id: str,
        name: str,
        description: str = "",
    ) -> dict[str, Any]:
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                """
                INSERT OR IGNORE INTO topics (id, name, description, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (topic_id, name, description, now),
            )
            await db.commit()
            cursor = await db.execute("SELECT * FROM topics WHERE id = ?", (topic_id,))
            row = await cursor.fetchone()
            return dict(row) if row else {}

    # --- Documents CRUD ---
    async def get_document(self, doc_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def get_document_by_hash(
        self,
        topic_id: str,
        file_hash: str,
    ) -> dict[str, Any] | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM documents WHERE topic_id = ? AND file_hash = ?",
                (topic_id, file_hash),
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def upsert_document(
        self,
        doc_id: str,
        topic_id: str,
        title: str,
        source_type: str,
        file_hash: str,
        source_uri: str = "",
        status: str = "pending",
    ) -> None:
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO documents (
                    id, topic_id, title, source_type, source_uri,
                    file_hash, status, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    topic_id = excluded.topic_id,
                    title = excluded.title,
                    source_type = excluded.source_type,
                    source_uri = excluded.source_uri,
                    file_hash = excluded.file_hash,
                    status = excluded.status,
                    error_message = '',
                    updated_at = excluded.updated_at
                """,
                (doc_id, topic_id, title, source_type, source_uri, file_hash, status, now),
            )
            await db.commit()

    async def save_parsed_and_card(
        self,
        doc_id: str,
        parsed_markdown: str,
        paper_card_json: str,
        status: str = "parsed",
    ) -> None:
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE documents
                SET parsed_markdown = ?, paper_card_json = ?, status = ?, updated_at = ?
                WHERE id = ?
                """,
                (parsed_markdown, paper_card_json, status, now, doc_id),
            )
            await db.commit()

    async def update_document_status(
        self,
        doc_id: str,
        status: str,
        error_message: str = "",
    ) -> None:
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE documents
                SET status = ?, error_message = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, error_message, now, doc_id),
            )
            await db.commit()

    async def list_documents_by_topic(self, topic_id: str) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM documents WHERE topic_id = ? ORDER BY updated_at DESC",
                (topic_id,),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]