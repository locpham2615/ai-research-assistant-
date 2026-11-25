import re

from ai_research_assistant.harness.llm_router import TaskType
from ai_research_assistant.harness.schema_guard import SchemaGuard
from ai_research_assistant.rag.context_builder import ContextBuilder
from ai_research_assistant.retrieval.hybrid_retriever import HybridRetriever
from ai_research_assistant.retrieval.reranker import CrossEncoderReranker
from ai_research_assistant.retrieval.schemas import (
    Citation,
    GroundedAnswer,
    RAGAnswerDraft,
    RetrievedChunk,
)

_CITATION_BRACKET_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

_RAG_SYSTEM_PROMPT = (
    "You are an academic research assistant answering strictly from the provided sources.\n"
    "Rules:\n"
    "1. Answer ONLY using facts explicitly supported by the numbered sources [1], [2], ...\n"
    "2. Respond in the same language as the user's question.\n"
    "3. Attach inline citations like [1] or [1, 2] to every factual claim. "
    "Never cite a number that is not in the provided sources.\n"
    "4. If the sources do not contain enough evidence to answer the question, "
    "set is_sufficient_evidence to false and state the gap in abstain_reason.\n"
    "Return valid JSON matching the requested schema."
)


def _make_snippet(text: str, max_chars: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def verify_and_extract_citations(
    answer_text: str,
    sources: dict[int, RetrievedChunk],
) -> tuple[str, list[Citation], list[int]]:
    """
    Đối chiếu các chỉ số trích dẫn [k] trong câu trả lời với danh sách nguồn thật.
    - Loại bỏ mọi chỉ số ngoài phạm vi (hallucinated citations).
    - Trả về (câu trả lời đã làm sạch, danh sách Citation hợp lệ, danh sách chỉ số ảo đã xóa).
    """
    valid_seen: list[int] = []
    invalid_seen: list[int] = []

    def _replace_match(match: re.Match[str]) -> str:
        raw_nums = [int(tok.strip()) for tok in match.group(1).split(",")]
        kept: list[int] = []
        for num in raw_nums:
            if num in sources:
                kept.append(num)
                if num not in valid_seen:
                    valid_seen.append(num)
            elif num not in invalid_seen:
                invalid_seen.append(num)

        if not kept:
            return ""
        deduped_kept = list(dict.fromkeys(kept))
        return "[" + ", ".join(str(n) for n in deduped_kept) + "]"

    cleaned = _CITATION_BRACKET_RE.sub(_replace_match, answer_text)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    citations = [
        Citation(
            index=idx,
            doc_id=sources[idx].doc_id,
            title=sources[idx].title or sources[idx].doc_id,
            section=sources[idx].section or "General",
            chunk_index=sources[idx].chunk_index,
            snippet=_make_snippet(sources[idx].text),
        )
        for idx in valid_seen
    ]
    return cleaned, citations, invalid_seen


class CitationGroundedRAG:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        schema_guard: SchemaGuard,
        context_builder: ContextBuilder | None = None,
        max_context_chars: int = 7500,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.schema_guard = schema_guard
        self.context_builder = context_builder or ContextBuilder(
            max_context_chars=max_context_chars
        )

    async def answer(
        self,
        query: str,
        topic_id: str | None = None,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> GroundedAnswer:
        candidates, trace = await self.retriever.retrieve(
            query=query,
            topic_id=topic_id,
            doc_ids=doc_ids,
        )

        rerank_query = trace.rewritten_query.search_query or query
        reranked = await self.reranker.rerank(
            query=rerank_query,
            chunks=candidates,
            top_k=top_k,
        )
        trace.reranked_chunks = reranked

        # Chốt chặn 1: Không có đoạn bằng chứng nào vượt ngưỡng rerank
        if not reranked:
            return GroundedAnswer(
                query=query,
                answer=(
                    "Không tìm thấy bằng chứng đủ liên quan trong kho tài liệu "
                    "để trả lời câu hỏi này."
                ),
                citations=[],
                is_grounded=False,
                abstain_reason="no_relevant_evidence",
                retrieval_trace=trace,
            )

        built = self.context_builder.build(reranked)
        user_prompt = (
            f"Question:\n{query.strip()}\n\n"
            f"Sources:\n{built.formatted_context}"
        )

        draft = await self.schema_guard.complete_structured(
            task_type=TaskType.RAG_ANSWER,
            messages=[
                {"role": "system", "content": _RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_model=RAGAnswerDraft,
        )

        # Chốt chặn 2: LLM xác nhận nguồn không đủ bằng chứng
        if not draft.is_sufficient_evidence:
            reason = draft.abstain_reason.strip() or "insufficient_evidence"
            return GroundedAnswer(
                query=query,
                answer=(
                    draft.answer.strip()
                    or "Các tài liệu hiện có không đủ thông tin để kết luận."
                ),
                citations=[],
                is_grounded=False,
                abstain_reason=reason,
                retrieval_trace=trace,
            )

        # Chốt chặn 3: Hậu kiểm trích dẫn [k]
        cleaned_answer, citations, invalid_removed = verify_and_extract_citations(
            draft.answer,
            built.sources,
        )

        if not citations:
            return GroundedAnswer(
                query=query,
                answer=cleaned_answer,
                citations=[],
                is_grounded=False,
                abstain_reason="no_valid_citations",
                invalid_citations_removed=invalid_removed,
                retrieval_trace=trace,
            )

        return GroundedAnswer(
            query=query,
            answer=cleaned_answer,
            citations=citations,
            is_grounded=True,
            invalid_citations_removed=invalid_removed,
            retrieval_trace=trace,
        )