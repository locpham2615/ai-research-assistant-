import logging
import re

from pydantic import BaseModel, Field

from ai_research_assistant.eval.schemas import EvalSample, JudgeScores
from ai_research_assistant.harness.llm_router import TaskType
from ai_research_assistant.harness.schema_guard import SchemaGuard
from ai_research_assistant.retrieval.schemas import GroundedAnswer

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict academic RAG evaluator. Score the assistant's answer on "
    "three criteria using an integer scale from 1 (worst) to 5 (best):\n"
    "1. faithfulness: Are all factual claims in the answer strictly supported "
    "by the Retrieved Evidence? (5 = fully supported, 1 = fabricated/contradicted)\n"
    "2. correctness: Does the answer accurately capture the key facts in the "
    "Reference Answer? (5 = completely accurate, 1 = wrong or misses core facts)\n"
    "3. relevancy: Does the answer directly address the User Question without "
    "off-topic filler? (5 = fully relevant, 1 = off-topic)\n"
    "Return valid JSON matching the requested schema with a concise rationale."
)


class JudgeRubricOutput(BaseModel):
    faithfulness: int = Field(ge=1, le=5)
    correctness: int = Field(ge=1, le=5)
    relevancy: int = Field(ge=1, le=5)
    rationale: str = Field(default="")


def normalize_likert_score(score: int) -> float:
    """Chuẩn hóa điểm nguyên thang 1..5 về [0.0, 1.0]."""
    clamped = max(1, min(5, score))
    return (clamped - 1) / 4.0


def _compact(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _format_evidence(answer: GroundedAnswer, max_chars: int) -> str:
    entries: list[tuple[str, str]] = []
    if answer.retrieval_trace and answer.retrieval_trace.reranked_chunks:
        for idx, chunk in enumerate(answer.retrieval_trace.reranked_chunks, start=1):
            label = f"[{idx}] {chunk.section or 'General'}"
            entries.append((label, _compact(chunk.text)))
    elif answer.citations:
        for cit in answer.citations:
            label = f"[{cit.index}] {cit.section}"
            entries.append((label, _compact(cit.snippet)))

    if not entries:
        return "(No retrieved evidence)"

    blocks: list[str] = []
    used = 0
    for label, text in entries:
        if not text:
            continue
        block = f"{label}: {text}"
        sep = 1 if blocks else 0
        if used + sep + len(block) <= max_chars:
            blocks.append(block)
            used += sep + len(block)
            continue

        remaining = max_chars - used - sep - len(label) - 5
        if remaining > 40:
            blocks.append(f"{label}: {text[:remaining].rstrip()}...")
        break

    return "\n".join(blocks)


class RAGJudge:
    def __init__(
        self,
        schema_guard: SchemaGuard,
        max_evidence_chars: int = 4000,
    ) -> None:
        if max_evidence_chars <= 0:
            raise ValueError("max_evidence_chars must be positive")
        self.schema_guard = schema_guard
        self.max_evidence_chars = max_evidence_chars

    async def evaluate(
        self,
        sample: EvalSample,
        answer: GroundedAnswer,
    ) -> JudgeScores | None:
        if sample.should_abstain:
            return None

        if not answer.is_grounded:
            reason = answer.abstain_reason or "ungrounded"
            return JudgeScores(
                faithfulness=1.0,
                correctness=0.0,
                relevancy=0.0,
                rationale=f"False refusal: system abstained ({reason}).",
            )

        evidence_text = _format_evidence(answer, self.max_evidence_chars)
        user_prompt = (
            f"User Question:\n{sample.question.strip()}\n\n"
            f"Reference Answer:\n{sample.reference_answer.strip()}\n\n"
            f"Retrieved Evidence:\n{evidence_text}\n\n"
            f"Assistant Answer:\n{answer.answer.strip()}"
        )

        try:
            raw = await self.schema_guard.complete_structured(
                task_type=TaskType.EVAL_JUDGE,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=JudgeRubricOutput,
            )
            return JudgeScores(
                faithfulness=normalize_likert_score(raw.faithfulness),
                correctness=normalize_likert_score(raw.correctness),
                relevancy=normalize_likert_score(raw.relevancy),
                rationale=raw.rationale.strip(),
            )
        except Exception as exc:
            logger.warning(
                "RAGJudge failed on sample %s (%s). Skipping judge scores.",
                sample.sample_id,
                type(exc).__name__,
            )
            return None