import logging
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ArxivSearchProtocol(Protocol):
    async def search(self, query: str, max_results: int = 2) -> list[Any]: ...


class IngestArxivProtocol(Protocol):
    async def ingest_arxiv(
        self,
        arxiv_id: str,
        topic_id: str,
        download_dir: Path,
        force: bool = False,
    ) -> Any: ...


class ArxivDiscoveryResult(BaseModel):
    query: str
    discovered_arxiv_ids: list[str] = Field(default_factory=list)
    ingested_doc_ids: list[str] = Field(default_factory=list)
    failed_arxiv_ids: list[str] = Field(default_factory=list)
    note: str = ""


def _extract_arxiv_id(item: Any) -> str:
    if isinstance(item, dict):
        raw = item.get("arxiv_id") or item.get("id") or ""
    else:
        raw = getattr(item, "arxiv_id", None) or getattr(item, "id", None) or ""
    return str(raw).strip()


class ArxivDiscoveryTool:
    def __init__(
        self,
        arxiv_client: ArxivSearchProtocol,
        ingest_pipeline: IngestArxivProtocol,
        download_dir: Path = Path("data/pdfs"),
        max_papers: int = 2,
    ) -> None:
        if max_papers <= 0:
            raise ValueError("max_papers must be positive")
        self.arxiv_client = arxiv_client
        self.ingest_pipeline = ingest_pipeline
        self.download_dir = download_dir
        self.max_papers = max_papers

    async def discover_and_ingest(
        self,
        query: str,
        topic_id: str,
        max_papers: int | None = None,
    ) -> ArxivDiscoveryResult:
        cleaned = query.strip()
        limit = max_papers if max_papers is not None else self.max_papers
        if not cleaned or limit <= 0:
            return ArxivDiscoveryResult(query=cleaned, note="empty_query")

        try:
            papers = await self.arxiv_client.search(cleaned, max_results=limit)
        except Exception as exc:
            logger.warning(
                "ArxivDiscoveryTool search failed (%s): %s",
                type(exc).__name__,
                exc,
            )
            return ArxivDiscoveryResult(
                query=cleaned,
                note=f"search_error:{type(exc).__name__}",
            )

        arxiv_ids: list[str] = []
        for paper in papers[:limit]:
            aid = _extract_arxiv_id(paper)
            if aid and aid not in arxiv_ids:
                arxiv_ids.append(aid)

        if not arxiv_ids:
            return ArxivDiscoveryResult(query=cleaned, note="no_papers_found")

        self.download_dir.mkdir(parents=True, exist_ok=True)
        ingested: list[str] = []
        failed: list[str] = []

        for aid in arxiv_ids:
            try:
                res = await self.ingest_pipeline.ingest_arxiv(
                    aid,
                    topic_id=topic_id,
                    download_dir=self.download_dir,
                    force=False,
                )
                status = str(getattr(res, "status", ""))
                doc_id = str(getattr(res, "doc_id", ""))
                if status == "failed":
                    failed.append(aid)
                elif doc_id:
                    ingested.append(doc_id)
            except Exception as exc:
                logger.warning(
                    "ArxivDiscoveryTool failed to ingest %s (%s)",
                    aid,
                    type(exc).__name__,
                )
                failed.append(aid)

        return ArxivDiscoveryResult(
            query=cleaned,
            discovered_arxiv_ids=arxiv_ids,
            ingested_doc_ids=ingested,
            failed_arxiv_ids=failed,
            note="ok" if ingested else "no_papers_ingested",
        )