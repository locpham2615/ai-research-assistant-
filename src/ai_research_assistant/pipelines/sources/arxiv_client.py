import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

from ai_research_assistant.db.sqlite import SQLiteStore
from ai_research_assistant.pipelines.schemas import PaperMetadata

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _clean_arxiv_id(raw_id: str) -> str:
    """Chuẩn hóa arXiv ID từ URL hoặc chuỗi (vd: '2401.00001v1')."""
    cleaned = raw_id.strip()
    if "/abs/" in cleaned:
        cleaned = cleaned.split("/abs/")[-1]
    elif "/pdf/" in cleaned:
        cleaned = cleaned.split("/pdf/")[-1]
    cleaned = cleaned.removesuffix(".pdf")
    return cleaned.removeprefix("arXiv:").removeprefix("arxiv:")


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_arxiv_atom_xml(xml_text: str) -> list[PaperMetadata]:
    """Hàm thuần túy tách Atom XML của arXiv thành danh sách PaperMetadata."""
    root = ET.fromstring(xml_text)
    papers: list[PaperMetadata] = []

    for entry in root.findall("atom:entry", ATOM_NS):
        raw_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
        arxiv_id = _clean_arxiv_id(raw_id)
        if not arxiv_id:
            continue

        title = _normalize_whitespace(
            entry.findtext("atom:title", default="", namespaces=ATOM_NS)
        )
        abstract = _normalize_whitespace(
            entry.findtext("atom:summary", default="", namespaces=ATOM_NS)
        )
        published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)
        published_year = (
            int(published[:4])
            if len(published) >= 4 and published[:4].isdigit()
            else None
        )
        authors = [
            _normalize_whitespace(author.findtext("atom:name", default="", namespaces=ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
        ]
        authors = [a for a in authors if a]

        papers.append(
            PaperMetadata(
                doc_id=f"arxiv_{arxiv_id}",
                title=title or arxiv_id,
                authors=authors,
                published_year=published_year,
                source_type="arxiv",
                source_uri=f"https://arxiv.org/abs/{arxiv_id}",
                abstract=abstract,
            )
        )

    return papers


class ArxivClient:
    def __init__(
        self,
        sqlite_store: SQLiteStore | None = None,
        http_client: httpx.AsyncClient | None = None,
        cache_ttl_seconds: int = 21600,
    ) -> None:
        self.sqlite_store = sqlite_store
        self._external_client = http_client
        self.cache_ttl_seconds = cache_ttl_seconds

    async def _get_text(self, url: str, params: dict[str, str | int]) -> str:
        if self._external_client is not None:
            resp = await self._external_client.get(url, params=params, timeout=30.0)
            resp.raise_for_status()
            return resp.text

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.text

    async def search_papers(self, query: str, max_results: int = 5) -> list[PaperMetadata]:
        cache_key = hashlib.sha256(f"arxiv:search:{query}:{max_results}".encode()).hexdigest()
        if self.sqlite_store is not None:
            cached = await self.sqlite_store.get_api_cache(cache_key)
            if cached is not None:
                return [PaperMetadata.model_validate(item) for item in cached]

        xml_text = await self._get_text(
            ARXIV_API_URL,
            params={"search_query": f"all:{query}", "start": 0, "max_results": max_results},
        )
        papers = parse_arxiv_atom_xml(xml_text)

        if self.sqlite_store is not None:
            await self.sqlite_store.set_api_cache(
                cache_key,
                [p.model_dump() for p in papers],
                ttl_seconds=self.cache_ttl_seconds,
            )
        return papers

    async def fetch_by_id(self, arxiv_id: str) -> PaperMetadata | None:
        clean_id = _clean_arxiv_id(arxiv_id)
        cache_key = hashlib.sha256(f"arxiv:id:{clean_id}".encode()).hexdigest()
        if self.sqlite_store is not None:
            cached = await self.sqlite_store.get_api_cache(cache_key)
            if cached is not None:
                return PaperMetadata.model_validate(cached)

        xml_text = await self._get_text(ARXIV_API_URL, params={"id_list": clean_id})
        papers = parse_arxiv_atom_xml(xml_text)
        if not papers:
            return None

        paper = papers[0]
        if self.sqlite_store is not None:
            await self.sqlite_store.set_api_cache(
                cache_key,
                paper.model_dump(),
                ttl_seconds=self.cache_ttl_seconds,
            )
        return paper

    async def download_pdf(self, arxiv_id: str, dest_dir: Path) -> Path:
        clean_id = _clean_arxiv_id(arxiv_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = dest_dir / f"{clean_id.replace('/', '_')}.pdf"
        pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"

        if self._external_client is not None:
            resp = await self._external_client.get(pdf_url, timeout=60.0)
            resp.raise_for_status()
            pdf_path.write_bytes(resp.content)
            return pdf_path

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resp = await client.get(pdf_url)
            resp.raise_for_status()
            pdf_path.write_bytes(resp.content)
        return pdf_path