import logging

from ai_research_assistant.harness.llm_router import TaskType
from ai_research_assistant.harness.schema_guard import SchemaGuard
from ai_research_assistant.retrieval.schemas import QueryRewriteOutput, RewrittenQuery

logger = logging.getLogger(__name__)

_REWRITE_SYSTEM_PROMPT = (
    "You optimize user questions for searching an English academic paper corpus.\n"
    "Return JSON with:\n"
    "1. search_query: a concise, clear English academic search query.\n"
    "2. bm25_keywords: 3 to 6 core technical terms or phrases in English "
    "for exact keyword matching.\n"
    "Do not invent facts or answer the question."
)


class QueryRewriter:
    def __init__(self, schema_guard: SchemaGuard) -> None:
        self.schema_guard = schema_guard

    async def rewrite(self, query: str) -> RewrittenQuery:
        cleaned = query.strip()
        if not cleaned:
            return RewrittenQuery(
                original_query=query,
                search_query="",
                bm25_keywords=[],
                used_fallback=True,
            )

        try:
            output = await self.schema_guard.complete_structured(
                task_type=TaskType.QUERY_REWRITE,
                messages=[
                    {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": cleaned},
                ],
                response_model=QueryRewriteOutput,
            )
            search_query = output.search_query.strip()
            keywords = [kw.strip() for kw in output.bm25_keywords if kw.strip()]
            deduped_keywords = list(dict.fromkeys(keywords))[:6]

            if not search_query:
                raise ValueError("Empty search_query from LLM")

            return RewrittenQuery(
                original_query=query,
                search_query=search_query,
                bm25_keywords=deduped_keywords,
                used_fallback=False,
            )
        except Exception as exc:
            logger.warning(
                "Query rewrite failed (%s). Falling back to original query.",
                type(exc).__name__,
            )
            return RewrittenQuery(
                original_query=query,
                search_query=cleaned,
                bm25_keywords=[],
                used_fallback=True,
            )