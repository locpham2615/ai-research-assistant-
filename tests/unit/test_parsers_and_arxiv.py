from pathlib import Path
from typing import Any

from ai_research_assistant.db.sqlite import SQLiteStore
from ai_research_assistant.pipelines.parsers.document_parser import parse_grobid_tei_xml
from ai_research_assistant.pipelines.sources.arxiv_client import (
    ArxivClient,
    parse_arxiv_atom_xml,
)

SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is   All You Need</title>
    <summary>We propose the Transformer architecture.</summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
  </entry>
</feed>
"""

SAMPLE_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Sample Paper Title</title></titleStmt>
    </fileDesc>
    <profileDesc>
      <abstract><p>Short abstract summary.</p></abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head>1 Introduction</head>
        <p>First paragraph of intro.</p>
        <p>Second paragraph of intro.</p>
      </div>
      <div>
        <head>2 Method</head>
        <p>Core method description.</p>
      </div>
    </body>
    <back>
      <div type="references">
        <listBibl><biblStruct>Should be ignored</biblStruct></listBibl>
      </div>
    </back>
  </text>
</TEI>
"""


class FakeHttpResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class FakeHttpClient:
    def __init__(self, xml_text: str) -> None:
        self.xml_text = xml_text
        self.calls = 0

    async def get(self, url: str, **kwargs: Any) -> FakeHttpResponse:
        self.calls += 1
        return FakeHttpResponse(self.xml_text)


def test_parse_arxiv_atom_xml() -> None:
    papers = parse_arxiv_atom_xml(SAMPLE_ARXIV_XML)
    assert len(papers) == 1
    p = papers[0]
    assert p.doc_id == "arxiv_1706.03762v7"
    assert p.title == "Attention Is All You Need"
    assert p.published_year == 2017
    assert p.authors == ["Ashish Vaswani", "Noam Shazeer"]


async def test_arxiv_client_uses_sqlite_cache(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "test.db")
    await store.init_db()
    fake_http = FakeHttpClient(SAMPLE_ARXIV_XML)
    client = ArxivClient(sqlite_store=store, http_client=fake_http)  # type: ignore[arg-type]

    first = await client.fetch_by_id("1706.03762v7")
    second = await client.fetch_by_id("https://arxiv.org/abs/1706.03762v7")

    assert first is not None and second is not None
    assert first.doc_id == second.doc_id
    assert fake_http.calls == 1


def test_parse_grobid_tei_xml_excludes_references() -> None:
    doc = parse_grobid_tei_xml(SAMPLE_TEI_XML)
    assert doc.title == "Sample Paper Title"
    assert doc.abstract == "Short abstract summary."
    assert len(doc.sections) == 2
    assert doc.sections[0].title == "1 Introduction"
    assert "Second paragraph" in doc.sections[0].content
    assert "Should be ignored" not in doc.full_markdown