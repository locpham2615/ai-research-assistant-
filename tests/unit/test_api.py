import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from httpx import ASGITransport, AsyncClient

from ai_research_assistant.agent.memory import ChatTurn
from ai_research_assistant.agent.schemas import (
    AgentExecutionTrace,
    AgenticAnswer,
    AgentStepTrace,
)
from ai_research_assistant.api.container import AppContainer
from ai_research_assistant.api.main import create_app
from ai_research_assistant.config import Settings
from ai_research_assistant.db.sqlite import SQLiteStore
from ai_research_assistant.retrieval.schemas import Citation, GroundedAnswer


class FakeResearchAgent:
    def __init__(self) -> None:
        self.received_histories: list[list[ChatTurn] | None] = []

    async def answer(
        self,
        query: str,
        topic_id: str | None = None,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
        max_hops: int | None = None,
        allow_external_arxiv: bool | None = None,
        chat_history: list[ChatTurn] | None = None,
    ) -> AgenticAnswer:
        self.received_histories.append(chat_history)
        return AgenticAnswer(
            grounded_answer=GroundedAnswer(
                query=query,
                answer=f"Grounded answer for: {query} [1]",
                citations=[
                    Citation(
                        index=1,
                        doc_id="arxiv_2106.09685",
                        title="LoRA",
                        section="Method",
                        chunk_index=0,
                        snippet="LoRA adapts W_q and W_v.",
                    )
                ],
                is_grounded=True,
            ),
            agent_trace=AgentExecutionTrace(
                route="simple",
                hops_executed=1,
                used_fast_path=True,
                llm_calls=2,
                steps=[
                    AgentStepTrace(node="plan", hop=0, latency_ms=10.0, detail="simple"),
                    AgentStepTrace(node="synthesize", hop=1, latency_ms=20.0, detail="ok"),
                ],
            ),
        )

    async def stream_answer(
        self,
        query: str,
        topic_id: str | None = None,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
        max_hops: int | None = None,
        allow_external_arxiv: bool | None = None,
        chat_history: list[ChatTurn] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        result = await self.answer(
            query=query,
            topic_id=topic_id,
            doc_ids=doc_ids,
            top_k=top_k,
            max_hops=max_hops,
            allow_external_arxiv=allow_external_arxiv,
            chat_history=chat_history,
        )
        for step in result.agent_trace.steps:
            yield {"type": "step", "step": step}
        yield {"type": "final", "result": result}


class FakeSingleShotRAG:
    async def answer(
        self,
        query: str,
        topic_id: str | None = None,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> GroundedAnswer:
        return GroundedAnswer(
            query=query,
            answer="Single-shot answer [1]",
            citations=[
                Citation(
                    index=1,
                    doc_id="arxiv_1706.03762",
                    title="Attention Is All You Need",
                    section="3.2",
                    chunk_index=0,
                    snippet="Scaled dot-product attention.",
                )
            ],
            is_grounded=True,
        )


class FakeIngestPipeline:
    def __init__(self, db: SQLiteStore) -> None:
        self.db = db
        self.ingested_files: list[Path] = []

    async def ingest_arxiv(
        self,
        arxiv_id: str,
        topic_id: str,
        download_dir: Path,
        force: bool = False,
    ) -> Any:
        doc_id = f"arxiv_{arxiv_id}"
        await self.db.upsert_document(
            doc_id=doc_id,
            topic_id=topic_id,
            title=f"Paper {arxiv_id}",
            source_type="arxiv",
            file_hash=f"hash_{arxiv_id}",
            status="indexed",
        )
        return SimpleNamespace(
            doc_id=doc_id,
            status="indexed",
            chunks_count=12,
            skipped_reason="",
            error_message="",
        )

    async def ingest_file(
        self,
        file_path: Path,
        topic_id: str,
        force: bool = False,
    ) -> Any:
        self.ingested_files.append(file_path)
        doc_id = f"upload_{file_path.stem}"
        await self.db.upsert_document(
            doc_id=doc_id,
            topic_id=topic_id,
            title=file_path.name,
            source_type="upload",
            file_hash="hash_upload",
            status="indexed",
        )
        return SimpleNamespace(
            doc_id=doc_id,
            status="indexed",
            chunks_count=5,
            skipped_reason="",
            error_message="",
        )


def _parse_sse(raw_text: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in raw_text.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = ""
        data_line = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: ") :].strip()
            elif line.startswith("data: "):
                data_line = line[len("data: ") :].strip()
        if event_name and data_line:
            events.append((event_name, json.loads(data_line)))
    return events


async def _make_test_app(tmp_path: Path) -> tuple[Any, FakeResearchAgent, FakeIngestPipeline]:
    db = SQLiteStore(tmp_path / "api_test.db")
    await db.init_db()
    settings = Settings(
        sqlite_db_path=tmp_path / "api_test.db",
        upload_dir=tmp_path / "uploads",
        arxiv_download_dir=tmp_path / "pdfs",
    )
    agent = FakeResearchAgent()
    pipeline = FakeIngestPipeline(db)
    container = AppContainer(
        settings=settings,
        sqlite_store=db,
        research_agent=agent,
        rag_engine=FakeSingleShotRAG(),
        ingest_pipeline=pipeline,
    )
    app = create_app(container=container)
    app.state.container = container
    return app, agent, pipeline


async def test_sessions_and_multi_turn_chat_persists_history(tmp_path: Path) -> None:
    app, agent, _ = await _make_test_app(tmp_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Lượt 1: tự tạo session mới khi không truyền session_id
        r1 = await client.post(
            "/api/chat",
            json={"query": "What is LoRA?", "topic_id": "llm_eval_v1"},
        )
        assert r1.status_code == 200
        body1 = r1.json()
        sid = body1["session_id"]
        assert body1["is_grounded"] is True
        assert len(body1["citations"]) == 1
        assert agent.received_histories[0] is None

        # Lượt 2: truyền session_id -> tự nạp lịch sử lượt 1 làm chat_history
        r2 = await client.post(
            "/api/chat",
            json={"query": "How does it compare to QLoRA?", "session_id": sid},
        )
        assert r2.status_code == 200
        hist2 = agent.received_histories[1]
        assert hist2 is not None
        assert [t.role for t in hist2] == ["user", "assistant"]
        assert hist2[0].content == "What is LoRA?"

        # Đọc lại messages, rename & delete session
        msgs = await client.get(f"/api/sessions/{sid}/messages")
        assert msgs.status_code == 200
        assert len(msgs.json()) == 4

        renamed = await client.patch(f"/api/sessions/{sid}", json={"title": "LoRA Study"})
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "LoRA Study"

        deleted = await client.delete(f"/api/sessions/{sid}")
        assert deleted.status_code == 204
        assert (await client.get(f"/api/sessions/{sid}/messages")).status_code == 404


async def test_chat_stream_emits_session_steps_and_final_sse(tmp_path: Path) -> None:
    app, _, _ = await _make_test_app(tmp_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/chat/stream",
            json={"query": "Explain LoRA", "topic_id": "llm_eval_v1", "mode": "agentic"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse(resp.text)
        names = [name for name, _ in events]
        assert names == ["session", "step", "step", "final"]
        final_payload = events[-1][1]
        assert final_payload["is_grounded"] is True
        assert final_payload["agent_trace"]["hops_executed"] == 1

        # Kiểm tra chế độ single_shot cũng stream mượt
        resp_ss = await client.post(
            "/api/chat/stream",
            json={"query": "Explain Attention", "mode": "single_shot"},
        )
        events_ss = _parse_sse(resp_ss.text)
        assert [name for name, _ in events_ss] == ["session", "step", "final"]
        assert events_ss[-1][1]["mode"] == "single_shot"


async def test_topics_arxiv_ingest_and_pdf_upload(tmp_path: Path) -> None:
    app, _, pipeline = await _make_test_app(tmp_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        t_resp = await client.post(
            "/api/topics",
            json={"topic_id": "llm_eval_v1", "name": "LLM Eval"},
        )
        assert t_resp.status_code == 201

        ingest_resp = await client.post(
            "/api/documents/ingest-arxiv",
            json={"arxiv_id": "2106.09685", "topic_id": "llm_eval_v1"},
        )
        assert ingest_resp.status_code == 200
        assert ingest_resp.json()["chunks_count"] == 12

        # Chặn file không phải PDF
        bad_upload = await client.post(
            "/api/documents/upload",
            files={"file": ("notes.txt", b"hello", "text/plain")},
            data={"topic_id": "llm_eval_v1"},
        )
        assert bad_upload.status_code == 400

        # Upload PDF hợp lệ
        good_upload = await client.post(
            "/api/documents/upload",
            files={"file": ("paper.pdf", b"%PDF-1.4 dummy content", "application/pdf")},
            data={"topic_id": "llm_eval_v1"},
        )
        assert good_upload.status_code == 200
        assert len(pipeline.ingested_files) == 1

        docs = await client.get("/api/documents", params={"topic_id": "llm_eval_v1"})
        assert docs.status_code == 200
        assert len(docs.json()) == 2

        topics = await client.get("/api/topics")
        assert topics.status_code == 200
        assert topics.json()[0]["document_count"] == 2