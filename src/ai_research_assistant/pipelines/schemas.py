from typing import Literal

from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    doc_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    published_year: int | None = None
    source_type: Literal["arxiv", "upload"] = "upload"
    source_uri: str = ""
    abstract: str = ""


class ParsedSection(BaseModel):
    title: str
    content: str


class ParsedDocument(BaseModel):
    title: str = ""
    abstract: str = ""
    sections: list[ParsedSection] = Field(default_factory=list)
    full_markdown: str = ""


class PaperCard(BaseModel):
    problem_statement: str = Field(
        description="Core research problem addressed by the paper."
    )
    methodology: str = Field(
        description="Main method, architecture, or algorithm proposed."
    )
    key_contributions: list[str] = Field(
        default_factory=list,
        description="Bullet list of primary contributions.",
    )
    datasets_or_benchmarks: list[str] = Field(
        default_factory=list,
        description="Datasets or benchmarks used for evaluation.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Stated or apparent limitations of the approach.",
    )


class DocumentChunk(BaseModel):
    doc_id: str
    topic_id: str
    section: str
    chunk_index: int
    text: str
    text_hash: str


class IngestResult(BaseModel):
    doc_id: str
    topic_id: str
    status: Literal["indexed", "skipped", "failed"]
    chunks_count: int = 0
    skipped_reason: str = ""
    error_message: str = ""