from langgraph.graph import StateGraph

from ai_research_assistant.agent.schemas import EvidenceGrade, QueryPlan, SubQuery
from ai_research_assistant.agent.state import ResearchAgentState, create_initial_state
from ai_research_assistant.config import Settings
from ai_research_assistant.harness.llm_router import LLMRouter, TaskType


def test_query_plan_normalizes_route_caps_queries_and_deduplicates_ids() -> None:
    # 1 sub_query nhưng LLM gắn nhầm route='decompose' -> tự chuẩn hóa về 'simple'
    single = QueryPlan.model_validate(
        {
            "route": "decompose",
            "sub_queries": [
                {
                    "sub_query_id": "",
                    "search_query": "  scaled dot-product attention sqrt d_k  ",
                    "bm25_keywords": ["sqrt", "softmax", "sqrt", "", "k1", "k2", "k3", "k4", "k5"],
                }
            ],
        }
    )
    assert single.route == "simple"
    assert single.sub_queries[0].sub_query_id == "sq1"
    assert single.sub_queries[0].search_query == "scaled dot-product attention sqrt d_k"
    assert single.sub_queries[0].bm25_keywords == ["sqrt", "softmax", "k1", "k2", "k3", "k4"]

    # 4 sub_queries nhưng route='simple' -> cắt tối đa 3 và chuẩn hóa về 'decompose'
    multi = QueryPlan.model_validate(
        {
            "route": "simple",
            "sub_queries": [
                {"sub_query_id": "dup", "search_query": "LoRA target matrices"},
                {"sub_query_id": "dup", "search_query": "QLoRA target matrices"},
                {"sub_query_id": "dup", "search_query": "LoRA vs QLoRA memory"},
                {"sub_query_id": "dup", "search_query": "extra ignored query"},
            ],
        }
    )
    assert multi.route == "decompose"
    assert len(multi.sub_queries) == 3
    assert [sq.sub_query_id for sq in multi.sub_queries] == ["dup", "sq2", "sq3"]


def test_evidence_grade_caps_follow_up_queries_and_cleans_aspects() -> None:
    grade = EvidenceGrade.model_validate(
        {
            "is_sufficient": False,
            "missing_aspects": ["  QLoRA adapter layers ", ""],
            "follow_up_queries": [
                {"search_query": "QLoRA all linear layers vs attention only"},
                {"search_query": "LoRA W_q W_v matrices"},
                {"search_query": "third extra query dropped"},
            ],
            "rationale": "Missing QLoRA details.",
        }
    )
    assert grade.missing_aspects == ["QLoRA adapter layers"]
    assert len(grade.follow_up_queries) == 2
    assert isinstance(grade.follow_up_queries[0], SubQuery)


def test_initial_state_and_langgraph_compile_smoke() -> None:
    state = create_initial_state(
        query="Compare LoRA and QLoRA",
        topic_id="llm_eval_v1",
        max_hops=2,
        top_k=6,
    )
    assert state["hop_count"] == 0
    assert state["allow_external_arxiv"] is False
    assert state["evidence_chunks"] == []

    graph = StateGraph(ResearchAgentState)
    graph.add_node("noop", lambda s: {"hop_count": s.get("hop_count", 0) + 1})
    graph.set_entry_point("noop")
    graph.set_finish_point("noop")
    compiled = graph.compile()
    result = compiled.invoke(state)
    assert result["hop_count"] == 1


def test_llm_router_routes_agent_tasks_to_local() -> None:
    router = LLMRouter(settings=Settings())
    assert router._resolve_provider(TaskType.AGENT_PLAN) == "local"
    assert router._resolve_provider(TaskType.AGENT_GRADE) == "local"