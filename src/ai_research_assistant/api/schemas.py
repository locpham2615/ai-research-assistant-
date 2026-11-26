from typing import Any, Literal

from pydantic import BaseModel, Field

ChatMode = Literal["agentic", "single_shot"]


class ServiceStatus(BaseModel):
    sqlite_ok: bool
    qdrant_ok: bool
    grobid_ok: bool
    local_llm_ok: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    services: ServiceStatus
    topics_count: int = 0
    documents_count: int = 0
    local_llm_model: str = ""


class TopicCreateRequest(BaseModel):
    topic_id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)


class TopicItem(BaseModel):
    id: str
    name: str
    description: str = ""
    created_at: float = 0.0
    document_count: int = 0


class DocumentItem(BaseModel):
    id: str
    topic_id: str
    title: str
    source_type: str
    source_uri: str = ""
    status: str
    error_message: str = ""
    updated_at: float = 0.0
    paper_card: dict[str, Any] | None = None


class IngestArxivRequest(BaseModel):
    arxiv_id: str = Field(min_length=4, max_length=64)
    topic_id: str = Field(default="llm_eval_v1", min_length=1, max_length=80)
    force: bool = False


class IngestResponse(BaseModel):
    doc_id: str
    topic_id: str
    status: str
    chunks_count: int = 0
    note: str = ""


class SessionCreateRequest(BaseModel):
    topic_id: str | None = None
    title: str = Field(default="New Research Chat", max_length=120)


class SessionUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class SessionResponse(BaseModel):
    session_id: str
    topic_id: str | None = None
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class MessageResponse(BaseModel):
    message_id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    agent_trace: dict[str, Any] | None = None
    is_grounded: bool | None = None
    abstain_reason: str = ""
    created_at: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    topic_id: str | None = None
    doc_ids: list[str] | None = None
    mode: ChatMode = "agentic"
    allow_external_arxiv: bool = False
    max_hops: int | None = Field(default=None, ge=1, le=3)
    top_k: int | None = Field(default=None, ge=1, le=12)


class ChatResponse(BaseModel):
    session_id: str
    user_message_id: str
    assistant_message_id: str
    mode: ChatMode
    query: str
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    is_grounded: bool
    abstain_reason: str = ""
    invalid_citations_removed: list[int] = Field(default_factory=list)
    agent_trace: dict[str, Any] | None = None