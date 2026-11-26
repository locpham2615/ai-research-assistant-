import json
import re
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from ai_research_assistant.api.container import AppContainer, get_container
from ai_research_assistant.api.schemas import (
    DocumentItem,
    IngestArxivRequest,
    IngestResponse,
    TopicCreateRequest,
    TopicItem,
)

router = APIRouter(prefix="/api", tags=["documents"])

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


def _parse_paper_card(raw: str) -> dict[str, Any] | None:
    if not raw or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _row_to_document(row: dict[str, Any]) -> DocumentItem:
    return DocumentItem(
        id=str(row.get("id", "")),
        topic_id=str(row.get("topic_id", "")),
        title=str(row.get("title", "")),
        source_type=str(row.get("source_type", "")),
        source_uri=str(row.get("source_uri", "")),
        status=str(row.get("status", "")),
        error_message=str(row.get("error_message", "")),
        updated_at=float(row.get("updated_at", 0.0)),
        paper_card=_parse_paper_card(str(row.get("paper_card_json", ""))),
    )


@router.get("/topics", response_model=list[TopicItem])
async def list_topics(
    container: Annotated[AppContainer, Depends(get_container)],
) -> list[TopicItem]:
    rows = await container.sqlite_store.list_topics()
    return [TopicItem.model_validate(r) for r in rows]


@router.post("/topics", response_model=TopicItem, status_code=status.HTTP_201_CREATED)
async def create_topic(
    payload: TopicCreateRequest,
    container: Annotated[AppContainer, Depends(get_container)],
) -> TopicItem:
    topic = await container.sqlite_store.create_or_get_topic(
        topic_id=payload.topic_id.strip(),
        name=payload.name.strip(),
        description=payload.description.strip(),
    )
    return TopicItem(
        id=str(topic.get("id", payload.topic_id)),
        name=str(topic.get("name", payload.name)),
        description=str(topic.get("description", "")),
        created_at=float(topic.get("created_at", 0.0)),
        document_count=0,
    )


@router.get("/documents", response_model=list[DocumentItem])
async def list_documents(
    container: Annotated[AppContainer, Depends(get_container)],
    topic_id: Annotated[str | None, Query()] = None,
) -> list[DocumentItem]:
    if topic_id:
        rows = await container.sqlite_store.list_documents_by_topic(topic_id)
    else:
        rows = await container.sqlite_store.list_all_documents()
    return [_row_to_document(r) for r in rows]


@router.post("/documents/ingest-arxiv", response_model=IngestResponse)
async def ingest_arxiv_document(
    payload: IngestArxivRequest,
    container: Annotated[AppContainer, Depends(get_container)],
) -> IngestResponse:
    arxiv_id = payload.arxiv_id.strip()
    topic_id = payload.topic_id.strip()
    download_dir: Path = container.settings.arxiv_download_dir
    download_dir.mkdir(parents=True, exist_ok=True)

    await container.sqlite_store.create_or_get_topic(
        topic_id=topic_id,
        name=topic_id,
    )
    res = await container.ingest_pipeline.ingest_arxiv(
        arxiv_id,
        topic_id=topic_id,
        download_dir=download_dir,
        force=payload.force,
    )
    note = str(
        getattr(res, "skipped_reason", "")
        or getattr(res, "error_message", "")
        or "ok"
    )
    return IngestResponse(
        doc_id=str(getattr(res, "doc_id", f"arxiv_{arxiv_id}")),
        topic_id=topic_id,
        status=str(getattr(res, "status", "unknown")),
        chunks_count=int(getattr(res, "chunks_count", 0)),
        note=note,
    )


@router.post("/documents/upload", response_model=IngestResponse)
async def upload_pdf_document(
    file: Annotated[UploadFile, File(...)],
    container: Annotated[AppContainer, Depends(get_container)],
    topic_id: Annotated[str, Form()] = "llm_eval_v1",
    force: Annotated[bool, Form()] = False,
) -> IngestResponse:
    original_name = (file.filename or "document.pdf").strip()
    if not original_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files (.pdf) are supported.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="PDF exceeds 50 MB size limit.",
        )

    safe_name = _SAFE_FILENAME_RE.sub("_", Path(original_name).name)
    upload_dir: Path = container.settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    dest_path.write_bytes(content)

    cleaned_topic = topic_id.strip() or "llm_eval_v1"
    await container.sqlite_store.create_or_get_topic(
        topic_id=cleaned_topic,
        name=cleaned_topic,
    )

    res = await container.ingest_pipeline.ingest_file(
        dest_path,
        topic_id=cleaned_topic,
        force=force,
    )
    note = str(
        getattr(res, "skipped_reason", "")
        or getattr(res, "error_message", "")
        or "ok"
    )
    return IngestResponse(
        doc_id=str(getattr(res, "doc_id", dest_path.stem)),
        topic_id=cleaned_topic,
        status=str(getattr(res, "status", "unknown")),
        chunks_count=int(getattr(res, "chunks_count", 0)),
        note=note,
    )