import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from ai_research_assistant.agent.memory import ChatTurn
from ai_research_assistant.agent.schemas import AgenticAnswer, AgentStepTrace
from ai_research_assistant.api.container import AppContainer, get_container
from ai_research_assistant.api.schemas import ChatRequest, ChatResponse
from ai_research_assistant.retrieval.schemas import GroundedAnswer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _sse_event(event: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


def _make_session_title(query: str) -> str:
    cleaned = " ".join(query.strip().split())
    if len(cleaned) <= 60:
        return cleaned or "New Research Chat"
    return f"{cleaned[:57].rstrip()}..."


async def _prepare_turn(
    payload: ChatRequest,
    container: AppContainer,
) -> tuple[str, str | None, str, list[ChatTurn]]:
    db = container.sqlite_store
    if payload.session_id:
        session = await db.get_session(payload.session_id)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {payload.session_id}",
            )
        session_id = session.session_id
        effective_topic = payload.topic_id or session.topic_id
        history = await db.get_recent_chat_turns(
            session_id=session_id,
            max_turns=container.settings.chat_history_max_turns,
        )
    else:
        effective_topic = payload.topic_id
        session = await db.create_session(
            topic_id=effective_topic,
            title=_make_session_title(payload.query),
        )
        session_id = session.session_id
        history = []

    user_msg = await db.add_message(
        session_id=session_id,
        role="user",
        content=payload.query,
    )
    return session_id, effective_topic, user_msg.message_id, history


async def _persist_assistant_result(
    container: AppContainer,
    session_id: str,
    user_message_id: str,
    payload: ChatRequest,
    grounded: GroundedAnswer,
    agent_trace: dict[str, Any] | None,
) -> ChatResponse:
    citations = [c.model_dump() for c in grounded.citations]
    assistant_msg = await container.sqlite_store.add_message(
        session_id=session_id,
        role="assistant",
        content=grounded.answer,
        citations=citations,
        agent_trace=agent_trace,
        is_grounded=grounded.is_grounded,
        abstain_reason=grounded.abstain_reason,
    )
    return ChatResponse(
        session_id=session_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_msg.message_id,
        mode=payload.mode,
        query=payload.query,
        answer=grounded.answer,
        citations=citations,
        is_grounded=grounded.is_grounded,
        abstain_reason=grounded.abstain_reason,
        invalid_citations_removed=list(grounded.invalid_citations_removed),
        agent_trace=agent_trace,
    )


@router.post("", response_model=ChatResponse)
async def chat_once(
    payload: ChatRequest,
    container: Annotated[AppContainer, Depends(get_container)],
) -> ChatResponse:
    session_id, effective_topic, user_msg_id, history = await _prepare_turn(
        payload,
        container,
    )

    if payload.mode == "single_shot":
        grounded: GroundedAnswer = await container.rag_engine.answer(
            query=payload.query,
            topic_id=effective_topic,
            doc_ids=payload.doc_ids,
            top_k=payload.top_k,
        )
        return await _persist_assistant_result(
            container=container,
            session_id=session_id,
            user_message_id=user_msg_id,
            payload=payload,
            grounded=grounded,
            agent_trace=None,
        )

    agentic: AgenticAnswer = await container.research_agent.answer(
        query=payload.query,
        topic_id=effective_topic,
        doc_ids=payload.doc_ids,
        top_k=payload.top_k,
        max_hops=payload.max_hops,
        allow_external_arxiv=payload.allow_external_arxiv,
        chat_history=history or None,
    )
    return await _persist_assistant_result(
        container=container,
        session_id=session_id,
        user_message_id=user_msg_id,
        payload=payload,
        grounded=agentic.grounded_answer,
        agent_trace=agentic.agent_trace.model_dump(),
    )


@router.post("/stream")
async def chat_stream(
    payload: ChatRequest,
    container: Annotated[AppContainer, Depends(get_container)],
) -> StreamingResponse:
    session_id, effective_topic, user_msg_id, history = await _prepare_turn(
        payload,
        container,
    )

    async def event_generator() -> AsyncIterator[str]:
        yield _sse_event(
            "session",
            {
                "session_id": session_id,
                "user_message_id": user_msg_id,
                "mode": payload.mode,
            },
        )
        try:
            if payload.mode == "single_shot":
                start = time.perf_counter()
                grounded: GroundedAnswer = await container.rag_engine.answer(
                    query=payload.query,
                    topic_id=effective_topic,
                    doc_ids=payload.doc_ids,
                    top_k=payload.top_k,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                yield _sse_event(
                    "step",
                    AgentStepTrace(
                        node="single_shot_rag",
                        hop=1,
                        latency_ms=elapsed_ms,
                        detail=(
                            f"grounded={grounded.is_grounded}, "
                            f"citations={len(grounded.citations)}"
                        ),
                    ).model_dump(),
                )
                response = await _persist_assistant_result(
                    container=container,
                    session_id=session_id,
                    user_message_id=user_msg_id,
                    payload=payload,
                    grounded=grounded,
                    agent_trace=None,
                )
                yield _sse_event("final", response.model_dump())
                return

            final_agentic: AgenticAnswer | None = None
            async for item in container.research_agent.stream_answer(
                query=payload.query,
                topic_id=effective_topic,
                doc_ids=payload.doc_ids,
                top_k=payload.top_k,
                max_hops=payload.max_hops,
                allow_external_arxiv=payload.allow_external_arxiv,
                chat_history=history or None,
            ):
                if item.get("type") == "step":
                    step = item["step"]
                    step_data = (
                        step.model_dump()
                        if isinstance(step, AgentStepTrace)
                        else dict(step)
                    )
                    yield _sse_event("step", step_data)
                elif item.get("type") == "final":
                    final_agentic = item["result"]

            if final_agentic is None:
                raise RuntimeError("Agent stream finished without final answer.")

            response = await _persist_assistant_result(
                container=container,
                session_id=session_id,
                user_message_id=user_msg_id,
                payload=payload,
                grounded=final_agentic.grounded_answer,
                agent_trace=final_agentic.agent_trace.model_dump(),
            )
            yield _sse_event("final", response.model_dump())
        except Exception as exc:
            logger.exception("Error while streaming chat response")
            yield _sse_event(
                "error",
                {"detail": f"{type(exc).__name__}: {exc}"},
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )