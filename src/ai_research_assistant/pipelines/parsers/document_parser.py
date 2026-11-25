import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import pymupdf  # type: ignore[import-untyped]

from ai_research_assistant.pipelines.schemas import ParsedDocument, ParsedSection

logger = logging.getLogger(__name__)

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
_REFERENCES_RE = re.compile(r"\n\s*(?:references|bibliography)\s*\n", re.IGNORECASE)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _build_markdown(title: str, abstract: str, sections: list[ParsedSection]) -> str:
    parts: list[str] = []
    if title:
        parts.append(f"# {title}")
    if abstract:
        parts.append(f"## Abstract\n\n{abstract}")
    for sec in sections:
        parts.append(f"## {sec.title}\n\n{sec.content}")
    return "\n\n".join(parts).strip()


def parse_grobid_tei_xml(tei_xml: str) -> ParsedDocument:
    """
    Tách TEI XML trả về từ GROBID thành ParsedDocument theo từng section.
    Chỉ đọc <body>/<div>, tự động bỏ qua <back> (References/Bibliography).
    """
    root = ET.fromstring(tei_xml)

    title_el = root.find(".//tei:titleStmt/tei:title", TEI_NS)
    title = _clean_text("".join(title_el.itertext())) if title_el is not None else ""

    abstract_el = root.find(".//tei:profileDesc/tei:abstract", TEI_NS)
    abstract = ""
    if abstract_el is not None:
        abs_paras = [
            _clean_text("".join(p.itertext()))
            for p in abstract_el.findall(".//tei:p", TEI_NS)
        ]
        abstract = "\n\n".join(p for p in abs_paras if p)

    sections: list[ParsedSection] = []
    body_el = root.find(".//tei:text/tei:body", TEI_NS)
    if body_el is not None:
        for div in body_el.findall("tei:div", TEI_NS):
            head_el = div.find("tei:head", TEI_NS)
            sec_title = (
                _clean_text("".join(head_el.itertext()))
                if head_el is not None
                else "Untitled Section"
            )
            paras = [
                _clean_text("".join(p.itertext()))
                for p in div.findall("tei:p", TEI_NS)
            ]
            content = "\n\n".join(p for p in paras if p)
            if content:
                sections.append(ParsedSection(title=sec_title, content=content))

    return ParsedDocument(
        title=title,
        abstract=abstract,
        sections=sections,
        full_markdown=_build_markdown(title, abstract, sections),
    )


class GrobidParser:
    def __init__(
        self,
        grobid_url: str = "http://localhost:8070",
        http_client: httpx.AsyncClient | None = None,
        timeout_sec: float = 120.0,
    ) -> None:
        self.grobid_url = grobid_url.rstrip("/")
        self._external_client = http_client
        self.timeout_sec = timeout_sec

    async def parse_pdf(self, pdf_path: Path) -> ParsedDocument:
        with pymupdf.open(pdf_path) as doc:
            pages = [
                str(doc.load_page(i).get_text("text"))
                for i in range(doc.page_count)
            ]

        raw_text = "\n\n".join(pages).strip()
        # Cắt bỏ phần References ở cuối nếu phát hiện
        split_match = _REFERENCES_RE.split(raw_text, maxsplit=1)
        clean_body = split_match[0].strip()

        title = pdf_path.stem
        sections = [ParsedSection(title="Full Text", content=clean_body)] if clean_body else []
        return ParsedDocument(
            title=title,
            abstract="",
            sections=sections,
            full_markdown=_build_markdown(title, "", sections),
        )


class PyMuPDFParser:
    """Fallback parser dùng PyMuPDF khi GROBID không khả dụng hoặc cho file không chuẩn paper."""

    def parse_pdf(self, pdf_path: Path) -> ParsedDocument:
        with pymupdf.open(pdf_path) as doc:
            pages = [
                str(doc.load_page(i).get_text("text"))
                for i in range(doc.page_count)
            ]

        raw_text = "\n\n".join(pages).strip()
        # Cắt bỏ phần References ở cuối nếu phát hiện
        split_match = _REFERENCES_RE.split(raw_text, maxsplit=1)
        clean_body = split_match[0].strip()

        title = pdf_path.stem
        sections = [ParsedSection(title="Full Text", content=clean_body)] if clean_body else []
        return ParsedDocument(
            title=title,
            abstract="",
            sections=sections,
            full_markdown=_build_markdown(title, "", sections),
        )


class DocumentParser:
    """Facade thống nhất: hỗ trợ .pdf (GROBID ưu tiên, PyMuPDF fallback) và .md/.txt."""

    def __init__(
        self,
        grobid_parser: GrobidParser | None = None,
        pymupdf_parser: PyMuPDFParser | None = None,
        grobid_url: str = "http://localhost:8070",
    ) -> None:
        self.grobid_parser = grobid_parser or GrobidParser(grobid_url=grobid_url)
        self.pymupdf_parser = pymupdf_parser or PyMuPDFParser()

    async def parse_file(self, file_path: Path) -> ParsedDocument:
        suffix = file_path.suffix.lower()
        if suffix in {".md", ".txt"}:
            text = file_path.read_text(encoding="utf-8").strip()
            sections = [ParsedSection(title="Content", content=text)] if text else []
            return ParsedDocument(
                title=file_path.stem,
                abstract="",
                sections=sections,
                full_markdown=_build_markdown(file_path.stem, "", sections),
            )

        if suffix == ".pdf":
            try:
                return await self.grobid_parser.parse_pdf(file_path)
            except (httpx.HTTPError, ET.ParseError) as exc:
                logger.warning(
                    "GROBID failed (%s) for %s. Falling back to PyMuPDF.",
                    type(exc).__name__,
                    file_path.name,
                )
                return self.pymupdf_parser.parse_pdf(file_path)

        raise ValueError(f"Unsupported file format: {suffix}")