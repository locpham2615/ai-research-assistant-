import asyncio
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends

from ai_research_assistant.api.container import AppContainer, get_container
from ai_research_assistant.api.schemas import HealthResponse, ServiceStatus

router = APIRouter(prefix="/api", tags=["health"])


async def _probe_http(url: str, timeout_sec: float = 2.0) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            resp = await client.get(url)
            return resp.status_code < 400
    except Exception:
        return False


def _llm_health_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        trimmed = trimmed[:-3]
    return f"{trimmed}/health"


@router.get("/health", response_model=HealthResponse)
async def health_check(
    container: Annotated[AppContainer, Depends(get_container)],
) -> HealthResponse:
    settings = container.settings
    sqlite_ok = True
    topics_count = 0
    documents_count = 0
    try:
        topics_count, documents_count = (
            await container.sqlite_store.count_topics_and_documents()
        )
    except Exception:
        sqlite_ok = False

    qdrant_url = f"{settings.qdrant_url.rstrip('/')}/healthz"
    grobid_url = f"{settings.grobid_url.rstrip('/')}/api/isalive"
    llm_url = _llm_health_url(settings.local_llm_base_url)

    qdrant_ok, grobid_ok, llm_ok = await asyncio.gather(
        _probe_http(qdrant_url),
        _probe_http(grobid_url),
        _probe_http(llm_url),
    )

    core_ok = sqlite_ok and qdrant_ok and llm_ok
    return HealthResponse(
        status="ok" if core_ok else "degraded",
        services=ServiceStatus(
            sqlite_ok=sqlite_ok,
            qdrant_ok=qdrant_ok,
            grobid_ok=grobid_ok,
            local_llm_ok=llm_ok,
        ),
        topics_count=topics_count,
        documents_count=documents_count,
        local_llm_model=settings.local_llm_model,
    )