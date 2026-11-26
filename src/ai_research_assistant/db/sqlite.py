import json
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from ai_research_assistant.agent.memory import (
    ChatMessage,
    ChatRole,
    ChatSession,
    ChatTurn,
)

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

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT UNIQUE NOT NULL,
    doc_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    section TEXT NOT NULL DEFAULT '',
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_topic_id ON chunks(topic_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    title,
    section,
    text,
    content='chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, title, section, text)
    VALUES (new.id, new.title, new.section, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, title, section, text)
    VALUES ('delete', old.id, old.title, old.section, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, title, section, text)
    VALUES ('delete', old.id, old.title, old.section, old.text);
    INSERT INTO chunks_fts(rowid, title, section, text)
    VALUES (new.id, new.title, new.section, new.text);
END;

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    topic_id TEXT,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations_json TEXT NOT NULL DEFAULT '[]',
    agent_trace_json TEXT,
    is_grounded INTEGER,
    abstain_reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
    ON chat_messages(session_id, created_at);
"""

_FTS5_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def build_fts5_query(query: str, max_tokens: int = 24) -> str:
    """Chuyển chuỗi truy vấn tự do thành biểu thức FTS5 an toàn (OR-joined)."""
    tokens = _FTS5_TOKEN_RE.findall(query.lower())
    unique_tokens = list(dict.fromkeys(tokens))[:max_tokens]
    if not unique_tokens:
        return ""
    return " OR ".join(f'"{tok}"' for tok in unique_tokens)


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

    async def set_api_cache(
        self,
        query_hash: str,
        payload: Any,
        ttl_seconds: int = 21600,
    ) -> None:
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

    # --- Chunks & BM25 (FTS5) ---
    async def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        rows = [
            (
                c["chunk_id"],
                c["doc_id"],
                c["topic_id"],
                c.get("title", ""),
                c.get("section", ""),
                c["chunk_index"],
                c["text"],
                c["text_hash"],
            )
            for c in chunks
        ]
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, topic_id, title, section,
                    chunk_index, text, text_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    doc_id = excluded.doc_id,
                    topic_id = excluded.topic_id,
                    title = excluded.title,
                    section = excluded.section,
                    chunk_index = excluded.chunk_index,
                    text = excluded.text,
                    text_hash = excluded.text_hash
                """,
                rows,
            )
            await db.commit()

    async def delete_chunks_by_doc_id(self, doc_id: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            await db.commit()

    async def search_bm25(
        self,
        query: str,
        topic_id: str | None = None,
        doc_ids: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        fts_query = build_fts5_query(query)
        if not fts_query or limit <= 0:
            return []
        if doc_ids is not None and len(doc_ids) == 0:
            return []

        clauses = ["chunks_fts MATCH ?"]
        params: list[Any] = [fts_query]

        if topic_id is not None:
            clauses.append("c.topic_id = ?")
            params.append(topic_id)

        if doc_ids is not None:
            placeholders = ", ".join("?" for _ in doc_ids)
            clauses.append(f"c.doc_id IN ({placeholders})")
            params.extend(doc_ids)

        where_sql = " AND ".join(clauses)
        params.append(limit)

        # Trọng số cột: title=0.5, section=1.5, text=1.0
        # FTS5 bm25() trả số âm (càng âm càng khớp) -> đảo dấu thành lexical_score dương
        sql = f"""
            SELECT
                c.chunk_id,
                c.doc_id,
                c.topic_id,
                c.title,
                c.section,
                c.chunk_index,
                c.text,
                bm25(chunks_fts, 0.5, 1.5, 1.0) AS raw_bm25
            FROM chunks_fts
            JOIN chunks c ON chunks_fts.rowid = c.id
            WHERE {where_sql}
            ORDER BY raw_bm25 ASC
            LIMIT ?
        """

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(sql, params)
            rows = await cursor.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            raw_score = float(item.pop("raw_bm25"))
            item["lexical_score"] = -raw_score
            results.append(item)
        return results

    # --- Chat Sessions & Messages (Short-Term Session Memory) ---
    async def create_session(
        self,
        topic_id: str | None = None,
        title: str = "New Research Chat",
        session_id: str | None = None,
    ) -> ChatSession:
        sid = session_id or uuid.uuid4().hex
        cleaned_title = title.strip() or "New Research Chat"
        now = datetime.now(UTC).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO chat_sessions (session_id, topic_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (sid, topic_id, cleaned_title, now, now),
            )
            await db.commit()

        return ChatSession(
            session_id=sid,
            topic_id=topic_id,
            title=cleaned_title,
            created_at=now,
            updated_at=now,
            message_count=0,
        )

    async def get_session(self, session_id: str) -> ChatSession | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT
                    s.session_id,
                    s.topic_id,
                    s.title,
                    s.created_at,
                    s.updated_at,
                    COUNT(m.message_id) AS message_count
                FROM chat_sessions s
                LEFT JOIN chat_messages m ON m.session_id = s.session_id
                WHERE s.session_id = ?
                GROUP BY s.session_id
                """,
                (session_id,),
            )
            row = await cursor.fetchone()

        if row is None:
            return None
        return ChatSession(
            session_id=str(row["session_id"]),
            topic_id=str(row["topic_id"]) if row["topic_id"] is not None else None,
            title=str(row["title"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            message_count=int(row["message_count"]),
        )

    async def list_sessions(
        self,
        topic_id: str | None = None,
        limit: int = 50,
    ) -> list[ChatSession]:
        if limit <= 0:
            return []

        params: list[Any] = []
        where_clause = ""
        if topic_id is not None:
            where_clause = "WHERE s.topic_id = ?"
            params.append(topic_id)
        params.append(limit)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                f"""
                SELECT
                    s.session_id,
                    s.topic_id,
                    s.title,
                    s.created_at,
                    s.updated_at,
                    COUNT(m.message_id) AS message_count
                FROM chat_sessions s
                LEFT JOIN chat_messages m ON m.session_id = s.session_id
                {where_clause}
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC, s.rowid DESC
                LIMIT ?
                """,
                params,
            )
            rows = await cursor.fetchall()

        return [
            ChatSession(
                session_id=str(row["session_id"]),
                topic_id=str(row["topic_id"]) if row["topic_id"] is not None else None,
                title=str(row["title"]),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
                message_count=int(row["message_count"]),
            )
            for row in rows
        ]

    async def update_session_title(self, session_id: str, title: str) -> bool:
        cleaned_title = title.strip()
        if not cleaned_title:
            raise ValueError("Session title must not be empty")
        now = datetime.now(UTC).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                UPDATE chat_sessions
                SET title = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (cleaned_title, now, session_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def delete_session(self, session_id: str) -> bool:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            cursor = await db.execute(
                "DELETE FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def add_message(
        self,
        session_id: str,
        role: ChatRole,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        agent_trace: dict[str, Any] | None = None,
        is_grounded: bool | None = None,
        abstain_reason: str = "",
        message_id: str | None = None,
    ) -> ChatMessage:
        cleaned_content = content.strip()
        if not cleaned_content:
            raise ValueError("Message content must not be empty")

        mid = message_id or uuid.uuid4().hex
        now = datetime.now(UTC).isoformat()
        citations_list = citations or []
        citations_json = json.dumps(citations_list, ensure_ascii=False)
        trace_json = (
            json.dumps(agent_trace, ensure_ascii=False)
            if agent_trace is not None
            else None
        )
        grounded_int = None if is_grounded is None else int(is_grounded)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            exists_cur = await db.execute(
                "SELECT 1 FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            )
            if await exists_cur.fetchone() is None:
                raise ValueError(f"Chat session not found: {session_id}")

            await db.execute(
                """
                INSERT INTO chat_messages (
                    message_id, session_id, role, content,
                    citations_json, agent_trace_json, is_grounded,
                    abstain_reason, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mid,
                    session_id,
                    role,
                    cleaned_content,
                    citations_json,
                    trace_json,
                    grounded_int,
                    abstain_reason,
                    now,
                ),
            )
            await db.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            await db.commit()

        return ChatMessage(
            message_id=mid,
            session_id=session_id,
            role=role,
            content=cleaned_content,
            citations=citations_list,
            agent_trace=agent_trace,
            is_grounded=is_grounded,
            abstain_reason=abstain_reason,
            created_at=now,
        )

    async def get_session_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[ChatMessage]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if limit is not None and limit > 0:
                cursor = await db.execute(
                    """
                    SELECT * FROM (
                        SELECT
                            rowid AS seq_id,
                            message_id, session_id, role, content,
                            citations_json, agent_trace_json, is_grounded,
                            abstain_reason, created_at
                        FROM chat_messages
                        WHERE session_id = ?
                        ORDER BY created_at DESC, rowid DESC
                        LIMIT ?
                    ) sub
                    ORDER BY created_at ASC, seq_id ASC
                    """,
                    (session_id, limit),
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT
                        message_id, session_id, role, content,
                        citations_json, agent_trace_json, is_grounded,
                        abstain_reason, created_at
                    FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY created_at ASC, rowid ASC
                    """,
                    (session_id,),
                )
            rows = await cursor.fetchall()

        messages: list[ChatMessage] = []
        for row in rows:
            raw_citations = json.loads(row["citations_json"]) if row["citations_json"] else []
            raw_trace = (
                json.loads(row["agent_trace_json"])
                if row["agent_trace_json"] is not None
                else None
            )
            raw_grounded = row["is_grounded"]
            role: ChatRole = "user" if str(row["role"]) == "user" else "assistant"
            messages.append(
                ChatMessage(
                    message_id=str(row["message_id"]),
                    session_id=str(row["session_id"]),
                    role=role,
                    content=str(row["content"]),
                    citations=raw_citations,
                    agent_trace=raw_trace,
                    is_grounded=None if raw_grounded is None else bool(raw_grounded),
                    abstain_reason=str(row["abstain_reason"] or ""),
                    created_at=str(row["created_at"]),
                )
            )
        return messages

    async def get_recent_chat_turns(
        self,
        session_id: str,
        max_turns: int = 4,
    ) -> list[ChatTurn]:
        if max_turns <= 0:
            return []
        messages = await self.get_session_messages(
            session_id=session_id,
            limit=max_turns * 2,
        )
        return [ChatTurn(role=m.role, content=m.content) for m in messages]
    
    async def list_topics(self) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT
                    t.id,
                    t.name,
                    t.description,
                    t.created_at,
                    COUNT(d.id) AS document_count
                FROM topics t
                LEFT JOIN documents d ON d.topic_id = t.id
                GROUP BY t.id
                ORDER BY t.created_at DESC
                """
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def list_all_documents(self, limit: int = 200) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM documents ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def count_topics_and_documents(self) -> tuple[int, int]:
        async with aiosqlite.connect(self.db_path) as db:
            cur_t = await db.execute("SELECT COUNT(*) FROM topics")
            cur_d = await db.execute("SELECT COUNT(*) FROM documents")
            row_t = await cur_t.fetchone()
            row_d = await cur_d.fetchone()
            return (
                int(row_t[0]) if row_t else 0,
                int(row_d[0]) if row_d else 0,
            )