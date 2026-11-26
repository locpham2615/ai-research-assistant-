import logging

from ai_research_assistant.agent.schemas import QueryPlan, SubQuery
from ai_research_assistant.harness.llm_router import TaskType
from ai_research_assistant.harness.schema_guard import SchemaGuard

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM_PROMPT = (
    "You are a strict academic research query planner for a local paper repository.\n"
    "Analyze the user's question (Vietnamese or English) and build a compact retrieval plan.\n"
    "Rules:\n"
    "1. DEFAULT to route='simple' (EXACTLY 1 sub-query). Even if the question asks "
    "multiple facts, numbers, steps, or causes about ONE paper/method/system, you MUST "
    "keep route='simple' with 1 sub-query.\n"
    "2. Use route='decompose' (2 sub-queries; 3 ONLY if 3 distinct papers are named) "
    "STRICTLY when the question explicitly compares, contrasts, or connects TWO OR MORE "
    "distinct papers/methods/systems (e.g., LoRA vs QLoRA, Self-RAG vs RAG, "
    "Self-RAG with vLLM, ReAct vs Toolformer, Self-RAG vs GraphRAG).\n"
    "3. Translate every search_query into crisp academic English and keep all exact "
    "entity names, model names, and numbers intact (never drop an entity to make a "
    "query easier to match).\n"
    "4. Each sub-query must have 3 to 6 exact technical bm25_keywords and a short "
    "target_aspect (max 6 words).\n"
    "5. Keep reasoning under 15 words. Return valid JSON matching the schema."
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