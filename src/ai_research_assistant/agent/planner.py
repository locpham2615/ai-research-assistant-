import logging

from ai_research_assistant.agent.schemas import QueryPlan, SubQuery
from ai_research_assistant.harness.llm_router import TaskType
from ai_research_assistant.harness.schema_guard import SchemaGuard

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM_PROMPT = (
    "You are an academic research query planner for a local paper repository.\n"
    "Analyze the user's question (which may be in Vietnamese or English) and build "
    "an optimal retrieval plan.\n"
    "Rules:\n"
    "1. Translate and normalize every search_query into crisp academic English.\n"
    "2. Use route='simple' (exactly 1 sub-query) for single-paper, single-concept, "
    "or straightforward factoid questions.\n"
    "3. Use route='decompose' (2 to 3 independent sub-queries) ONLY when the "
    "question compares/synthesizes multiple papers or methods (e.g., LoRA vs QLoRA, "
    "Self-RAG vs GraphRAG, vLLM in Self-RAG) or asks multiple distinct aspects "
    "that require separate evidence.\n"
    "4. Each sub-query must target ONE specific paper, method, or aspect with:\n"
    "   - search_query: focused English query.\n"
    "   - bm25_keywords: 3 to 6 exact technical terms, acronyms, symbols, or "
    "method names.\n"
    "   - target_aspect: short label of the fact this sub-query must retrieve.\n"
    "5. Do not answer the question. Return valid JSON matching the schema."
)


def _build_fallback_plan(query: str, reason: str) -> QueryPlan:
    cleaned = query.strip()
    return QueryPlan(
        route="simple",
        sub_queries=[
            SubQuery(
                sub_query_id="sq1",
                search_query=cleaned,
                bm25_keywords=[],
                target_aspect="original_query",
            )
        ],
        reasoning=reason,
    )


class QueryPlanner:
    def __init__(
        self,
        schema_guard: SchemaGuard,
        max_sub_queries: int = 3,
    ) -> None:
        if max_sub_queries <= 0:
            raise ValueError("max_sub_queries must be positive")
        self.schema_guard = schema_guard
        self.max_sub_queries = min(max_sub_queries, 3)

    async def plan(self, query: str) -> QueryPlan:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Query must not be empty")

        user_prompt = (
            f"User Question:\n{cleaned}\n\n"
            f"Produce a QueryPlan with at most {self.max_sub_queries} sub-queries."
        )

        try:
            plan = await self.schema_guard.complete_structured(
                task_type=TaskType.AGENT_PLAN,
                messages=[
                    {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=QueryPlan,
            )
            if len(plan.sub_queries) > self.max_sub_queries:
                capped = plan.sub_queries[: self.max_sub_queries]
                return QueryPlan(
                    route="decompose" if len(capped) > 1 else "simple",
                    sub_queries=capped,
                    reasoning=plan.reasoning,
                )
            return plan
        except Exception as exc:
            logger.warning(
                "QueryPlanner failed (%s). Falling back to single-query plan.",
                type(exc).__name__,
            )
            return _build_fallback_plan(
                cleaned,
                reason=f"fallback:{type(exc).__name__}",
            )