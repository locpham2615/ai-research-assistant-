import hashlib
import logging
import uuid
from pathlib import Path
from typing import Literal

from qdrant_client.models import PointStruct

from ai_research_assistant.db.sqlite import SQLiteStore
from ai_research_assistant.db.vector_store import VectorStore
from ai_research_assistant.harness.llm_router import TaskType
from ai_research_assistant.harness.schema_guard import SchemaGuard
from ai_research_assistant.pipelines.chunkers.section_chunker import SectionChunker
from ai_research_assistant.pipelines.embedders.text_embedder import TextEmbedder
from ai_research_assistant.pipelines.parsers.document_parser import DocumentParser
from ai_research_assistant.pipelines.schemas import (
    IngestResult,
    PaperCard,
    ParsedDocument,
)
from ai_research_assistant.pipelines.sources.arxiv_client import ArxivClient

logger = logging.getLogger(__name__)


def _sha256_file(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _make_point_id(doc_id: str, chunk_index: int) -> str:
    """Qdrant yêu cầu Point ID là UUID hợp lệ hoặc số nguyên dương."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{chunk_index}"))


def _build_card_excerpt(doc: ParsedDocument, max_chars: int = 6000) -> str:
    """Lấy phần tiêu đề, abstract và đầu các mục chính để vừa context 4096."""
    parts: list[str] = []
    if doc.title:
        parts.append(f"Title: {doc.title}")
    if doc.abstract:
        parts.append(f"Abstract:\n{doc.abstract}")
    for sec in doc.sections:
        parts.append(f"Section - {sec.title}:\n{sec.content}")
    combined = "\n\n".join(parts)
    return combined[:max_chars]


class IngestPipeline:
    def __init__(
        self,
        sqlite_store: SQLiteStore,
        vector_store: VectorStore,
        parser: DocumentParser,
        chunker: SectionChunker,
        embedder: TextEmbedder,
        schema_guard: SchemaGuard,
        arxiv_client: ArxivClient | None = None,
    ) -> None:
        self.sqlite_store = sqlite_store
        self.vector_store = vector_store
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.schema_guard = schema_guard
        self.arxiv_client = arxiv_client or ArxivClient(sqlite_store=sqlite_store)

    async def ingest_file(
        self,
        file_path: Path,
        topic_id: str,
        doc_id: str | None = None,
        title: str | None = None,
        source_type: Literal["arxiv", "upload"] = "upload",
        source_uri: str = "",
        force: bool = False,
    ) -> IngestResult:
        file_hash = _sha256_file(file_path)
        resolved_doc_id = doc_id or f"file_{file_hash[:12]}"
        resolved_title = title or file_path.stem

        await self.sqlite_store.create_or_get_topic(topic_id, name=topic_id)

        existing_by_id = await self.sqlite_store.get_document(resolved_doc_id)
        existing_by_hash = await self.sqlite_store.get_document_by_hash(topic_id, file_hash)

        if not force:
            if (
                existing_by_id
                and existing_by_id["file_hash"] == file_hash
                and existing_by_id["status"] == "indexed"
            ):
                return IngestResult(
                    doc_id=resolved_doc_id,
                    topic_id=topic_id,
                    status="skipped",
                    skipped_reason="already_indexed",
                )
            if existing_by_hash and existing_by_hash["status"] == "indexed":
                return IngestResult(
                    doc_id=existing_by_hash["id"],
                    topic_id=topic_id,
                    status="skipped",
                    skipped_reason="duplicate_hash",
                )

        # Delete-then-Upsert nếu tài liệu cũ bị cập nhật hoặc chạy lại với force=True
        if existing_by_id is not None:
            self.vector_store.delete_by_doc_id(resolved_doc_id)

        await self.sqlite_store.upsert_document(
            doc_id=resolved_doc_id,
            topic_id=topic_id,
            title=resolved_title,
            source_type=source_type,
            source_uri=source_uri or str(file_path),
            file_hash=file_hash,
            status="pending",
        )

        try:
            # 1. Parse tài liệu
            parsed_doc = await self.parser.parse_file(file_path)
            final_title = parsed_doc.title or resolved_title

            # 2. Trích xuất PaperCard qua SchemaGuard (local llama-server)
            excerpt = _build_card_excerpt(parsed_doc)
            paper_card = await self.schema_guard.complete_structured(
                task_type=TaskType.EXTRACT_CARD,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract a concise structured PaperCard from the following "
                            "research paper excerpt."
                        ),
                    },
                    {"role": "user", "content": excerpt},
                ],
                response_model=PaperCard,
            )

            await self.sqlite_store.save_parsed_and_card(
                doc_id=resolved_doc_id,
                parsed_markdown=parsed_doc.full_markdown,
                paper_card_json=paper_card.model_dump_json(),
                status="parsed",
            )

            # 3. Cắt khúc & tính Embedding (có SQLite cache)
            chunks = self.chunker.chunk_document(
                parsed_doc,
                doc_id=resolved_doc_id,
                topic_id=topic_id,
            )
            vectors = await self.embedder.embed_chunks(chunks)

            # 4. Đẩy vào Qdrant
            points = [
                PointStruct(
                    id=_make_point_id(chunk.doc_id, chunk.chunk_index),
                    vector=vector,
                    payload={
                        "doc_id": chunk.doc_id,
                        "topic_id": chunk.topic_id,
                        "title": final_title,
                        "section": chunk.section,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "text_hash": chunk.text_hash,
                    },
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            self.vector_store.upsert_chunks(points)

            # 5. Đánh dấu hoàn tất
            await self.sqlite_store.update_document_status(resolved_doc_id, "indexed")
            return IngestResult(
                doc_id=resolved_doc_id,
                topic_id=topic_id,
                status="indexed",
                chunks_count=len(chunks),
            )

        except Exception as exc:
            logger.exception("Failed to ingest document %s", resolved_doc_id)
            await self.sqlite_store.update_document_status(
                resolved_doc_id,
                status="failed",
                error_message=str(exc),
            )
            return IngestResult(
                doc_id=resolved_doc_id,
                topic_id=topic_id,
                status="failed",
                error_message=str(exc),
            )

    async def ingest_arxiv(
        self,
        arxiv_id: str,
        topic_id: str,
        download_dir: Path,
        force: bool = False,
    ) -> IngestResult:
        metadata = await self.arxiv_client.fetch_by_id(arxiv_id)
        if metadata is None:
            return IngestResult(
                doc_id=f"arxiv_{arxiv_id}",
                topic_id=topic_id,
                status="failed",
                error_message=f"arXiv paper not found: {arxiv_id}",
            )

        pdf_path = await self.arxiv_client.download_pdf(arxiv_id, download_dir)
        return await self.ingest_file(
            file_path=pdf_path,
            topic_id=topic_id,
            doc_id=metadata.doc_id,
            title=metadata.title,
            source_type="arxiv",
            source_uri=metadata.source_uri,
            force=force,
        )