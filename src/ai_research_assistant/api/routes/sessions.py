from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from ai_research_assistant.api.container import AppContainer, get_container
from ai_research_assistant.api.schemas import (
    MessageResponse,
    SessionCreateRequest,
    SessionResponse,
    SessionUpdateRequest,
)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionResponse])
async def list_sessions(
    container: Annotated[AppContainer, Depends(get_container)],
    topic_id: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[SessionResponse]:
    sessions = await container.sqlite_store.list_sessions(
        topic_id=topic_id,
        limit=limit,
    )
    return [SessionResponse.model_validate(s.model_dump()) for s in sessions]


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: SessionCreateRequest,
    container: Annotated[AppContainer, Depends(get_container)],
) -> SessionResponse:
    session = await container.sqlite_store.create_session(
        topic_id=payload.topic_id,
        title=payload.title,
    )
    return SessionResponse.model_validate(session.model_dump())


@router.patch("/{session_id}", response_model=SessionResponse)
async def rename_session(
    session_id: str,
    payload: SessionUpdateRequest,
    container: Annotated[AppContainer, Depends(get_container)],
) -> SessionResponse:
    updated = await container.sqlite_store.update_session_title(
        session_id=session_id,
        title=payload.title,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    session = await container.sqlite_store.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    return SessionResponse.model_validate(session.model_dump())


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    container: Annotated[AppContainer, Depends(get_container)],
) -> Response:
    deleted = await container.sqlite_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{session_id}/messages", response_model=list[MessageResponse])
async def get_session_messages(
    session_id: str,
    container: Annotated[AppContainer, Depends(get_container)],
) -> list[MessageResponse]:
    session = await container.sqlite_store.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    messages = await container.sqlite_store.get_session_messages(session_id)
    return [MessageResponse.model_validate(m.model_dump()) for m in messages]