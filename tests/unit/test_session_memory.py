from pathlib import Path
from typing import Any

import pytest

from ai_research_assistant.agent.evidence_pool import EvidencePool
from ai_research_assistant.agent.grader import EvidenceGrader
from ai_research_assistant.agent.memory import ChatTurn, format_chat_history
from ai_research_assistant.agent.planner import QueryPlanner
from ai_research_assistant.agent.research_agent import ResearchAgent
from ai_research_assistant.agent.schemas import QueryPlan, SubQuery
from ai_research_assistant.db.sqlite import SQLiteStore
from ai_research_assistant.retrieval.reranker import CrossEncoderReranker
from ai_research_assistant.retrieval.schemas import (
    RAGAnswerDraft,
    RetrievalTrace,
    RetrievedChunk,
    RewrittenQuery,
)


def test_format_chat_history_enforces_turns_and_chars_budget() -> None:
    assert format_chat_history(None) == ""
    assert format_chat_history([]) == ""

    turns = [
        ChatTurn(role="user", content="Turn 1 question about Transformer"),
        ChatTurn(role="assistant", content="Turn 1 answer about self-attention"),
        ChatTurn(role="user", content="Turn 2 question about LoRA"),
        ChatTurn(role="assistant", content="Turn 2 answer about low-rank adapters"),
        ChatTurn(role="user", content="Turn 3 question about QLoRA"),
        ChatTurn(role="assistant", content="Turn 3 answer about NF4 quantization"),
    ]

    # Chỉ giữ 2 lượt gần nhất (Turn 2 và Turn 3) theo đúng thứ tự thời gian
    formatted = format_chat_history(turns, max_turns=2, max_chars=1500)
    assert "Turn 1" not in formatted
    assert formatted.index("Turn 2 question") < formatted.index("Turn 3 answer")

    # Khóa trần ký tự rất nhỏ -> ưu tiên giữ tin nhắn mới nhất không vượt ngân sách
    tight = format_chat_history(turns, max_turns=4, max_chars=65)
    assert len(tight) <= 65
    assert "Turn 3 answer" in tight


async def test_sqlite_session_and_message_lifecycle_with_cascade_delete(
    tmp_path: Path,
) -> None:
    db = SQLiteStore(tmp_path / "memory_test.db")
    await db.init_db()

    s1 = await db.create_session(topic_id="llm_eval_v1", title="LoRA Deep Dive")
    s2 = await db.create_session(topic_id="other_topic", title="Vision Papers")

    assert s1.message_count == 0
    assert (await db.get_session(s1.session_id)) is not None
    assert [s.session_id for s in await db.list_sessions(topic_id="llm_eval_v1")] == [
        s1.session_id
    ]
    assert len(await db.list_sessions()) == 2

    assert await db.update_session_title(s1.session_id, "LoRA vs QLoRA") is True
    updated = await db.get_session(s1.session_id)
    assert updated is not None
    assert updated.title == "LoRA vs QLoRA"

    await db.add_message(s1.session_id, role="user", content="What is LoRA?")
    await db.add_message(
        s1.session_id,
        role="assistant",
        content="LoRA adapts attention weights with low-rank matrices [1].",
        citations=[{"index": 1, "doc_id": "arxiv_2106.09685"}],
        agent_trace={"route": "simple", "hops_executed": 1},
        is_grounded=True,
    )
    await db.add_message(s1.session_id, role="user", content="How does it differ from QLoRA?")

    messages = await db.get_session_messages(s1.session_id)
    assert [m.role for m in messages] == ["user", "assistant", "user"]
    assert messages[1].citations == [{"index": 1, "doc_id": "arxiv_2106.09685"}]
    assert messages[1].agent_trace == {"route": "simple", "hops_executed": 1}
    assert messages[1].is_grounded is True

    # Lấy 1 lượt gần nhất -> 2 tin nhắn cuối đúng thứ tự thời gian
    recent = await db.get_recent_chat_turns(s1.session_id, max_turns=1)
    assert [t.role for t in recent] == ["assistant", "user"]
    assert "QLoRA" in recent[1].content

    refreshed = await db.get_session(s1.session_id)
    assert refreshed is not None
    assert refreshed.message_count == 3

    # Xóa session phải tự cascade xóa sạch chat_messages
    assert await db.delete_session(s1.session_id) is True
    assert await db.get_session(s1.session_id) is None
    assert await db.get_session_messages(s1.session_id) == []
    assert (await db.get_session(s2.session_id)) is not None

    with pytest.raises(ValueError, match="not found"):
        await db.add_message("nonexistent", role="user", content="hello")


async def test_query_planner_and_agent_forward_chat_history() -> None:
    captured_user_prompts: list[str] = []

    class CapturingGuard:
        async def complete_structured(self, **kwargs: Any) -> Any:
            model = kwargs["response_model"]
            messages = kwargs["messages"]
            if model is QueryPlan:
                captured_user_prompts.append(messages[-1]["content"])
                return QueryPlan(
                    route="decompose",
                    sub_queries=[
                        SubQuery(sub_query_id="sq1", search_query="LoRA weight matrices"),
                        SubQuery(sub_query_id="sq2", search_query="QLoRA NF4 linear layers"),
                    ],
                )
            if model is RAGAnswerDraft:
                return RAGAnswerDraft(
                    answer="LoRA targets W_q/W_v [1], whereas QLoRA targets all linear layers [2].",
                    is_sufficient_evidence=True,
                )
            raise AssertionError(f"Unexpected model: {model}")

    class TwoDocRetriever:
        async def retrieve(
            self,
            query: str,
            topic_id: str | None = None,
            doc_ids: list[str] | None = None,
            top_k: int | None = None,
            rewritten_query: RewrittenQuery | None = None,
        ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
            doc_id = "arxiv_2305.14314" if "QLoRA" in query else "arxiv_2106.09685"
            chunk = RetrievedChunk(
                chunk_id=f"{doc_id}_0",
                doc_id=doc_id,
                topic_id="llm_eval_v1",
                title=doc_id,
                section="Method",
                chunk_index=0,
                text=f"Evidence for {query}",
                rrf_score=0.03,
            )
            trace = RetrievalTrace(
                rewritten_query=rewritten_query
                or RewrittenQuery(original_query=query, search_query=query, bm25_keywords=[]),
                dense_candidates=1,
                bm25_candidates=1,
                fused_candidates=1,
            )
            return [chunk], trace

    guard = CapturingGuard()
    planner = QueryPlanner(schema_guard=guard)  # type: ignore[arg-type]

    # Không có lịch sử -> không chèn block Conversation History
    await planner.plan("What is LoRA?")
    assert "Conversation History" not in captured_user_prompts[-1]

    # Có lịch sử -> chèn block Conversation History qua ResearchAgent.answer
    reranker = CrossEncoderReranker(
        top_k=6,
        min_score=-3.0,
        score_fn=lambda q, docs: [4.0 for _ in docs],
    )
    agent = ResearchAgent(
        planner=planner,
        evidence_pool=EvidencePool(retriever=TwoDocRetriever(), reranker=reranker),
        grader=EvidenceGrader(schema_guard=guard),  # type: ignore[arg-type]
        schema_guard=guard,  # type: ignore[arg-type]
    )

    history = [
        ChatTurn(role="user", content="LoRA hoạt động như thế nào?"),
        ChatTurn(
            role="assistant",
            content="LoRA đóng băng trọng số gốc và thêm ma trận hạng thấp.",
        ),
    ]
    result = await agent.answer(
        query="Nó khác gì QLoRA?",
        topic_id="llm_eval_v1",
        chat_history=history,
    )

    assert "Conversation History" in captured_user_prompts[-1]
    assert "LoRA hoạt động như thế nào?" in captured_user_prompts[-1]
    assert result.grounded_answer.is_grounded is True
    assert result.agent_trace.route == "decompose"
    assert len(result.grounded_answer.citations) == 2