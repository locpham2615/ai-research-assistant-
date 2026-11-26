import logging
import re

from pydantic import BaseModel

from ai_research_assistant.agent.schemas import (
    AgentNextAction,
    AgentRoute,
    EvidenceGrade,
    QueryPlan,
    SubQuery,
)
from ai_research_assistant.harness.llm_router import TaskType
from ai_research_assistant.harness.schema_guard import SchemaGuard
from ai_research_assistant.retrieval.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")

_GRADER_SYSTEM_PROMPT = (
    "You are an academic research evidence grader for a multi-hop RAG system.\n"
    "Inspect the user's question, planned aspects, and retrieved paper excerpts.\n"
    "Decide whether the current evidence covers all required papers, concepts, "
    "or factual details.\n"
    "Rules:\n"
    "1. Set is_sufficient=true if the excerpts contain enough facts to answer "
    "every aspect of the question.\n"
    "2. Set is_sufficient=false ONLY if a specific paper, comparison side, "
    "number, or technical mechanism is clearly missing.\n"
    "3. When is_sufficient=false, list concrete missing_aspects and generate "
    "1 to 2 targeted English follow_up_queries (with 3-6 exact bm25_keywords) "
    "that are DIFFERENT from Already Executed Queries.\n"
    "4. Do not answer the question. Return valid JSON matching the schema."
)


class GradeDecision(BaseModel):
    next_action: AgentNextAction
    grade: EvidenceGrade
    used_fast_path: bool = False
    llm_called: bool = False


def _norm_query(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def _format_evidence_preview(chunks: list[RetrievedChunk], max_chars: int) -> str:
    if not chunks:
        return "(No evidence retrieved)"

    blocks: list[str] = []
    used = 0
    for idx, chunk in enumerate(chunks, start=1):
        title = chunk.title.strip() or chunk.doc_id
        section = chunk.section.strip() or "General"
        score = f"{chunk.rerank_score:.2f}" if chunk.rerank_score is not None else "n/a"
        header = f"[{idx}] {title} ({chunk.doc_id}) | {section} | score={score}"
        body = _WHITESPACE_RE.sub(" ", chunk.text).strip()
        if not body:
            continue

        block = f"{header}\n{body}"
        sep = 2 if blocks else 0
        if used + sep + len(block) <= max_chars:
            blocks.append(block)
            used += sep + len(block)
            continue

        remaining = max_chars - used - sep - len(header) - 5
        if remaining > 60:
            blocks.append(f"{header}\n{body[:remaining].rstrip()}...")
        break

    return "\n\n".join(blocks) if blocks else "(No evidence retrieved)"


def _deduplicate_follow_ups(
    candidates: list[SubQuery],
    executed_queries: list[SubQuery],
    next_hop: int,
) -> list[SubQuery]:
    seen = {_norm_query(sq.search_query) for sq in executed_queries}
    fresh: list[SubQuery] = []

    for sq in candidates:
        norm = _norm_query(sq.search_query)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        fresh.append(
            sq.model_copy(update={"sub_query_id": f"h{next_hop}_sq{len(fresh) + 1}"})
        )
        if len(fresh) >= 2:
            break

    return fresh


class EvidenceGrader:
    def __init__(
        self,
        schema_guard: SchemaGuard,
        fast_path_min_score: float = 2.0,
        fast_path_min_chunks: int = 2,
        max_evidence_chars: int = 3500,
    ) -> None:
        if fast_path_min_chunks <= 0:
            raise ValueError("fast_path_min_chunks must be positive")
        if max_evidence_chars <= 0:
            raise ValueError("max_evidence_chars must be positive")
        self.schema_guard = schema_guard
        self.fast_path_min_score = fast_path_min_score
        self.fast_path_min_chunks = fast_path_min_chunks
        self.max_evidence_chars = max_evidence_chars

    def _qualifies_fast_path(
        self,
        route: AgentRoute,
        chunks: list[RetrievedChunk],
    ) -> bool:
        if route != "simple":
            return False
        strong_hits = sum(
            1
            for c in chunks
            if c.rerank_score is not None and c.rerank_score >= self.fast_path_min_score
        )
        return strong_hits >= self.fast_path_min_chunks

    async def grade(
        self,
        query: str,
        plan: QueryPlan | None,
        evidence_chunks: list[RetrievedChunk],
        executed_queries: list[SubQuery],
        hop_count: int,
        max_hops: int,
        allow_external_arxiv: bool = False,
        used_external_arxiv: bool = False,
    ) -> GradeDecision:
        route: AgentRoute = plan.route if plan is not None else "simple"

        # Chốt 1: Không có bằng chứng nào sau rerank
        if not evidence_chunks:
            if allow_external_arxiv and not used_external_arxiv and hop_count < max_hops:
                return GradeDecision(
                    next_action="arxiv_discovery",
                    grade=EvidenceGrade(
                        is_sufficient=False,
                        missing_aspects=["no_local_evidence"],
                        rationale="Local evidence empty; triggering arXiv discovery.",
                    ),
                )
            return GradeDecision(
                next_action="abstain",
                grade=EvidenceGrade(
                    is_sufficient=False,
                    missing_aspects=["no_relevant_evidence"],
                    rationale="No evidence passed rerank threshold.",
                ),
            )

        # Chốt 2: Đã chạm trần số hop -> tổng hợp ngay không tốn thêm lệnh LLM
        if hop_count >= max_hops:
            return GradeDecision(
                next_action="synthesize",
                grade=EvidenceGrade(
                    is_sufficient=True,
                    rationale=f"Reached hop budget ({hop_count}/{max_hops}).",
                ),
            )

        # Chốt 3: Fast-Path cho câu hỏi đơn có ít nhất N đoạn điểm rerank cao
        if self._qualifies_fast_path(route, evidence_chunks):
            return GradeDecision(
                next_action="synthesize",
                grade=EvidenceGrade(
                    is_sufficient=True,
                    rationale="Fast-path: strong rerank scores on simple query.",
                ),
                used_fast_path=True,
            )

        # Chốt 4: Gọi LLM Grader phản biện lỗ hổng & sinh truy vấn sửa sai
        planned_lines = (
            [
                f"- {sq.sub_query_id}: {sq.target_aspect or sq.search_query}"
                for sq in plan.sub_queries
            ]
            if plan is not None
            else []
        )
        executed_lines = [
            f"- {sq.search_query} (keywords: {', '.join(sq.bm25_keywords)})"
            for sq in executed_queries
        ]
        evidence_preview = _format_evidence_preview(
            evidence_chunks,
            max_chars=self.max_evidence_chars,
        )

        user_prompt = (
            f"User Question:\n{query.strip()}\n\n"
            f"Route: {route}\n"
            f"Planned Aspects:\n{chr(10).join(planned_lines) or '(none)'}\n\n"
            f"Already Executed Queries:\n{chr(10).join(executed_lines) or '(none)'}\n\n"
            f"Retrieved Evidence ({len(evidence_chunks)} excerpts):\n{evidence_preview}"
        )

        try:
            raw_grade = await self.schema_guard.complete_structured(
                task_type=TaskType.AGENT_GRADE,
                messages=[
                    {"role": "system", "content": _GRADER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=EvidenceGrade,
            )
        except Exception as exc:
            logger.warning(
                "EvidenceGrader failed (%s). Proceeding to synthesis.",
                type(exc).__name__,
            )
            return GradeDecision(
                next_action="synthesize",
                grade=EvidenceGrade(
                    is_sufficient=True,
                    rationale=f"fallback:{type(exc).__name__}",
                ),
                llm_called=True,
            )

        if raw_grade.is_sufficient:
            return GradeDecision(
                next_action="synthesize",
                grade=raw_grade,
                llm_called=True,
            )

        fresh_queries = _deduplicate_follow_ups(
            candidates=raw_grade.follow_up_queries,
            executed_queries=executed_queries,
            next_hop=hop_count + 1,
        )
        cleaned_grade = raw_grade.model_copy(update={"follow_up_queries": fresh_queries})

        if fresh_queries:
            return GradeDecision(
                next_action="refine",
                grade=cleaned_grade,
                llm_called=True,
            )

        return GradeDecision(
            next_action="synthesize",
            grade=cleaned_grade,
            llm_called=True,
        )