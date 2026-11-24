from ai_research_assistant.pipelines.chunkers.section_chunker import SectionChunker
from ai_research_assistant.pipelines.schemas import ParsedDocument, ParsedSection


def test_chunker_isolates_sections_and_sets_metadata() -> None:
    doc = ParsedDocument(
        title="Test Paper",
        abstract="Abstract content here.",
        sections=[
            ParsedSection(title="Introduction", content="Intro para 1.\n\nIntro para 2."),
            ParsedSection(title="Method", content="Method details."),
        ],
    )
    chunker = SectionChunker(max_chars=200, overlap_chars=30)
    chunks = chunker.chunk_document(doc, doc_id="doc_1", topic_id="topic_a")

    assert len(chunks) == 3
    assert [c.section for c in chunks] == ["Abstract", "Introduction", "Method"]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert all(c.doc_id == "doc_1" and c.topic_id == "topic_a" for c in chunks)
    assert all(len(c.text_hash) == 64 for c in chunks)


def test_chunker_splits_long_section_with_overlap() -> None:
    long_para = " ".join(f"Sentence number {i} explains attention weights." for i in range(15))
    doc = ParsedDocument(
        sections=[ParsedSection(title="Experiments", content=long_para)],
    )
    chunker = SectionChunker(max_chars=180, overlap_chars=40)
    chunks = chunker.chunk_document(doc, doc_id="doc_2", topic_id="topic_b")

    assert len(chunks) >= 3
    assert all(len(c.text) <= 180 for c in chunks)
    assert all(c.section == "Experiments" for c in chunks)