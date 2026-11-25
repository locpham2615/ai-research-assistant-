from pydantic import BaseModel, Field


class QueryRewriteOutput(BaseModel):
    """Cấu trúc JSON gọn nhẹ mà LLM trả về cho bước Query Rewrite."""

    search_query: str = Field(
        description=(
            "Concise academic search query in English, normalized from "
            "the user question."
        )
    )
    bm25_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "3 to 6 core technical terms or exact phrases for BM25 "
            "keyword matching."
        ),
    )


class RewrittenQuery(BaseModel):
    original_query: str
    search_query: str
    bm25_keywords: list[str] = Field(default_factory=list)
    used_fallback: bool = False


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    topic_id: str
    title: str = ""
    section: str = ""
    chunk_index: int
    text: str
    dense_score: float | None = None
    lexical_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None


class Citation(BaseModel):
    index: int
    doc_id: str
    title: str
    section: str
    chunk_index: int
    snippet: str


class RAGAnswerDraft(BaseModel):
    """Cấu trúc JSON mà LLM sinh ra trước khi qua bước hậu kiểm Citation."""

    answer: str = Field(
        description=(
            "Grounded answer using inline citations like [1], [2]. "
            "Leave concise explanation if evidence is insufficient."
        )
    )
    is_sufficient_evidence: bool = Field(
        description=(
            "True if the provided sources contain enough evidence to "
            "answer the question; otherwise False."
        )
    )
    abstain_reason: str = Field(
        default="",
        description="Reason for abstaining when is_sufficient_evidence is False.",
    )


class RetrievalTrace(BaseModel):
    rewritten_query: RewrittenQuery
    dense_candidates: int = 0
    bm25_candidates: int = 0
    fused_candidates: int = 0
    reranked_chunks: list[RetrievedChunk] = Field(default_factory=list)


class GroundedAnswer(BaseModel):
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    is_grounded: bool
    abstain_reason: str = ""
    invalid_citations_removed: list[int] = Field(default_factory=list)
    retrieval_trace: RetrievalTrace | None = None