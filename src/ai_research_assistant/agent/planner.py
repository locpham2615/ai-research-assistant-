import logging

from ai_research_assistant.agent.memory import ChatTurn, format_chat_history
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
    "3. If Conversation History is provided and the current question contains pronouns or "
    "follow-up references (e.g., 'it', 'that method', 'nó', 'phương pháp trên'), resolve "
    "them using the history so EVERY search_query is completely standalone and self-contained. "
    "Ignore history if the user introduces a new independent topic.\n"
    "4. Translate every search_query into crisp academic English and keep all exact "
    "entity names, model names, and numbers intact (never drop an entity to make a "
    "query easier to match).\n"
    "5. Each sub-query must have 3 to 6 exact technical bm25_keywords and a short "
    "target_aspect (max 6 words).\n"
    "6. Keep reasoning under 15 words. Return valid JSON matching the schema."
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
        max_history_turns: int = 4,
        max_history_chars: int = 1500,
    ) -> None:
        if max_sub_queries <= 0:
            raise ValueError("max_sub_queries must be positive")
        self.schema_guard = schema_guard
        self.max_sub_queries = min(max_sub_queries, 3)
        self.max_history_turns = max_history_turns
        self.max_history_chars = max_history_chars

    async def plan(
        self,
        query: str,
        chat_history: list[ChatTurn] | None = None,
    ) -> QueryPlan:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Query must not be empty")

        history_block = format_chat_history(
            chat_history,
            max_turns=self.max_history_turns,
            max_chars=self.max_history_chars,
        )
        if history_block:
            user_prompt = (
                f"Conversation History (for resolving follow-up references only):\n"
                f"{history_block}\n\n"
                f"User Question:\n{cleaned}\n\n"
                f"Produce a standalone QueryPlan with at most {self.max_sub_queries} sub-queries."
            )
        else:
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