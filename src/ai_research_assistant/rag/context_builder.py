from pydantic import BaseModel, Field

from ai_research_assistant.retrieval.schemas import RetrievedChunk


class BuiltContext(BaseModel):
    formatted_context: str = ""
    sources: dict[int, RetrievedChunk] = Field(default_factory=dict)
    total_chars: int = 0


def _format_source_block(index: int, chunk: RetrievedChunk, text: str) -> str:
    title = chunk.title.strip() or chunk.doc_id
    section = chunk.section.strip() or "General"
    header = f"[{index}] Title: {title} | Section: {section} | DocID: {chunk.doc_id}"
    return f"{header}\n{text}"


class ContextBuilder:
    def __init__(self, max_context_chars: int = 7500) -> None:
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive")
        self.max_context_chars = max_context_chars

    def build(self, chunks: list[RetrievedChunk]) -> BuiltContext:
        blocks: list[str] = []
        sources: dict[int, RetrievedChunk] = {}
        current_len = 0

        for idx, chunk in enumerate(chunks, start=1):
            body = chunk.text.strip()
            if not body:
                continue

            block = _format_source_block(idx, chunk, body)
            sep_len = 2 if blocks else 0

            if current_len + sep_len + len(block) <= self.max_context_chars:
                blocks.append(block)
                sources[idx] = chunk
                current_len += sep_len + len(block)
                continue

            # Nếu ngay chunk đầu tiên đã vượt ngân sách thì cắt gọn body để không mất hết ngữ cảnh
            if not blocks:
                empty_block = _format_source_block(idx, chunk, "")
                available = self.max_context_chars - len(empty_block) - 3
                if available > 0:
                    truncated = body[:available].rstrip() + "..."
                    first_block = _format_source_block(idx, chunk, truncated)
                    blocks.append(first_block)
                    sources[idx] = chunk
                    current_len = len(first_block)
            break

        formatted = "\n\n".join(blocks)
        return BuiltContext(
            formatted_context=formatted,
            sources=sources,
            total_chars=len(formatted),
        )