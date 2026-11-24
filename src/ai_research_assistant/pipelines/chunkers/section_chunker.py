import hashlib
import re

from ai_research_assistant.pipelines.schemas import DocumentChunk, ParsedDocument

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SectionChunker:
    def __init__(self, max_chars: int = 1000, overlap_chars: int = 150) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if overlap_chars < 0 or overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be >= 0 and < max_chars")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def _split_long_text(self, text: str) -> list[str]:
        """Tách một đoạn văn quá dài theo câu; nếu câu vẫn quá dài thì cắt cứng."""
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        units: list[str] = []

        for sent in sentences:
            if len(sent) <= self.max_chars:
                units.append(sent)
            else:
                step = self.max_chars - self.overlap_chars
                for start in range(0, len(sent), step):
                    piece = sent[start : start + self.max_chars].strip()
                    if piece:
                        units.append(piece)
                    if start + self.max_chars >= len(sent):
                        break
        return units

    def _chunk_section_text(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            return []

        units: list[str] = []
        for para in paragraphs:
            if len(para) <= self.max_chars:
                units.append(para)
            else:
                units.extend(self._split_long_text(para))

        chunks: list[str] = []
        current = ""

        for unit in units:
            if not current:
                current = unit
                continue

            candidate = f"{current}\n\n{unit}"
            if len(candidate) <= self.max_chars:
                current = candidate
            else:
                chunks.append(current)
                if self.overlap_chars > 0:
                    tail = current[-self.overlap_chars :].strip()
                    merged = f"{tail}\n\n{unit}" if tail else unit
                    current = merged if len(merged) <= self.max_chars else unit
                else:
                    current = unit

        if current:
            chunks.append(current)
        return chunks

    def chunk_document(
        self,
        doc: ParsedDocument,
        doc_id: str,
        topic_id: str,
    ) -> list[DocumentChunk]:
        section_pairs: list[tuple[str, str]] = []
        if doc.abstract.strip():
            section_pairs.append(("Abstract", doc.abstract.strip()))
        for sec in doc.sections:
            if sec.content.strip():
                section_pairs.append((sec.title.strip() or "Untitled Section", sec.content.strip()))

        result: list[DocumentChunk] = []
        chunk_idx = 0

        for section_title, section_text in section_pairs:
            for chunk_text in self._chunk_section_text(section_text):
                result.append(
                    DocumentChunk(
                        doc_id=doc_id,
                        topic_id=topic_id,
                        section=section_title,
                        chunk_index=chunk_idx,
                        text=chunk_text,
                        text_hash=_sha256_text(chunk_text),
                    )
                )
                chunk_idx += 1

        return result