from ai_research_assistant.agent.arxiv_tool import ArxivDiscoveryTool
from ai_research_assistant.agent.evidence_pool import (
    EvidencePool,
    RerankerProtocol,
    SubQueryRetrieverProtocol,
)
from ai_research_assistant.agent.grader import EvidenceGrader
from ai_research_assistant.agent.graph import build_research_graph
from ai_research_assistant.agent.memory import ChatTurn
from ai_research_assistant.agent.planner import QueryPlanner
from ai_research_assistant.agent.schemas import (
    AgentExecutionTrace,
    AgenticAnswer,
    AgentRoute,
    QueryPlan,
)
from ai_research_assistant.agent.state import create_initial_state
from ai_research_assistant.db.sqlite import SQLiteStore
from ai_research_assistant.db.vector_store import VectorStore
from ai_research_assistant.harness.schema_guard import SchemaGuard
from ai_research_assistant.pipelines.embedders.text_embedder import TextEmbedder
from ai_research_assistant.rag.context_builder import ContextBuilder
from ai_research_assistant.retrieval.hybrid_retriever import HybridRetriever
from ai_research_assistant.retrieval.reranker import CrossEncoderReranker
from ai_research_assistant.retrieval.schemas import GroundedAnswer


class ResearchAgent:
    def __init__(
        self,
        planner: QueryPlanner,
        evidence_pool: EvidencePool,
        grader: EvidenceGrader,
        schema_guard: SchemaGuard,
        context_builder: ContextBuilder | None = None,
        arxiv_tool: ArxivDiscoveryTool | None = None,
        max_hops: int = 2,
        default_top_k: int = 6,
        max_context_chars: int = 7500,
        allow_external_arxiv: bool = False,
    ) -> None:
        self.planner = planner
        self.evidence_pool = evidence_pool
        self.grader = grader
        self.schema_guard = schema_guard
        self.arxiv_tool = arxiv_tool
        self.max_hops = max(1, max_hops)
        self.default_top_k = max(1, default_top_k)
        self.allow_external_arxiv = allow_external_arxiv
        self._graph = build_research_graph(
            planner=planner,
            evidence_pool=evidence_pool,
            grader=grader,
            schema_guard=schema_guard,
            context_builder=context_builder,
            arxiv_tool=arxiv_tool,
            max_context_chars=max_context_chars,
        )

    @classmethod
    def from_components(
        cls,
        embedder: TextEmbedder,
        vector_store: VectorStore,
        sqlite_store: SQLiteStore,
        schema_guard: SchemaGuard,
        reranker_model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        retrieval_top_k: int = 20,
        rerank_min_score: float = -3.0,
        rrf_k: int = 60,
        max_hops: int = 2,
        max_sub_queries: int = 3,
        fast_path_min_score: float = 1.0,
        fast_path_min_chunks: int = 2,
        merged_top_k: int = 6,
        max_context_chars: int = 7500,
        allow_external_arxiv: bool = False,
        arxiv_tool: ArxivDiscoveryTool | None = None,
        chat_history_max_turns: int = 4,
        chat_history_max_chars: int = 1500,
    ) -> "ResearchAgent":
        retriever: SubQueryRetrieverProtocol = HybridRetriever(
            embedder=embedder,
            vector_store=vector_store,
            sqlite_store=sqlite_store,
            query_rewriter=None,
            retrieval_top_k=retrieval_top_k,
            rrf_k=rrf_k,
        )
        reranker: RerankerProtocol = CrossEncoderReranker(
            model_name=reranker_model_name,
            top_k=merged_top_k,
            min_score=rerank_min_score,
        )
        planner = QueryPlanner(
            schema_guard=schema_guard,
            max_sub_queries=max_sub_queries,
            max_history_turns=chat_history_max_turns,
            max_history_chars=chat_history_max_chars,
        )
        pool = EvidencePool(
            retriever=retriever,
            reranker=reranker,
            per_sub_query_top_k=max(3, merged_top_k // 2 + 1),
            merged_top_k=merged_top_k,
        )
        grader = EvidenceGrader(
            schema_guard=schema_guard,
            fast_path_min_score=fast_path_min_score,
            fast_path_min_chunks=fast_path_min_chunks,
        )
        return cls(
            planner=planner,
            evidence_pool=pool,
            grader=grader,
            schema_guard=schema_guard,
            arxiv_tool=arxiv_tool,
            max_hops=max_hops,
            default_top_k=merged_top_k,
            max_context_chars=max_context_chars,
            allow_external_arxiv=allow_external_arxiv,
        )

    async def answer(
        self,
        query: str,
        topic_id: str | None = None,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
        max_hops: int | None = None,
        allow_external_arxiv: bool | None = None,
        chat_history: list[ChatTurn] | None = None,
    ) -> AgenticAnswer:
        state = create_initial_state(
            query=query,
            topic_id=topic_id,
            doc_ids=doc_ids,
            max_hops=max_hops if max_hops is not None else self.max_hops,
            top_k=top_k if top_k is not None else self.default_top_k,
            allow_external_arxiv=(
                allow_external_arxiv
                if allow_external_arxiv is not None
                else self.allow_external_arxiv
            ),
            chat_history=chat_history,
        )
        final_state = await self._graph.ainvoke(state)

        grounded: GroundedAnswer | None = final_state.get("grounded_answer")
        if grounded is None:
            grounded = GroundedAnswer(
                query=query,
                answer="Không tìm thấy bằng chứng đủ liên quan trong kho tài liệu.",
                citations=[],
                is_grounded=False,
                abstain_reason="agent_no_output",
            )

        plan: QueryPlan | None = final_state.get("plan")
        route: AgentRoute = plan.route if plan is not None else "simple"
        trace = AgentExecutionTrace(
            route=route,
            hops_executed=int(final_state.get("hop_count", 0)),
            used_fast_path=bool(final_state.get("used_fast_path", False)),
            used_external_arxiv=bool(final_state.get("used_external_arxiv", False)),
            llm_calls=int(final_state.get("llm_calls", 0)),
            plan=plan,
            grades=list(final_state.get("grades", [])),
            steps=list(final_state.get("steps", [])),
        )
        return AgenticAnswer(
            grounded_answer=grounded,
            agent_trace=trace,
            pre_rerank_chunks=list(final_state.get("pre_rerank_chunks", [])),
        )