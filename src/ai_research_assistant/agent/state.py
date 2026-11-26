from typing import TypedDict

from ai_research_assistant.agent.memory import ChatTurn
from ai_research_assistant.agent.schemas import (
    AgentNextAction,
    AgentStepTrace,
    EvidenceGrade,
    QueryPlan,
    SubQuery,
)
from ai_research_assistant.retrieval.schemas import GroundedAnswer, RetrievedChunk


class ResearchAgentState(TypedDict, total=False):
    # Đầu vào phiên truy vấn
    query: str
    topic_id: str | None
    doc_ids: list[str] | None
    max_hops: int
    top_k: int
    allow_external_arxiv: bool
    chat_history: list[ChatTurn]

    # Kế hoạch và hàng đợi truy vấn con
    plan: QueryPlan | None
    pending_queries: list[SubQuery]
    executed_queries: list[SubQuery]

    # Kho bằng chứng tích lũy qua các hop
    pre_rerank_chunks: list[RetrievedChunk]
    evidence_chunks: list[RetrievedChunk]
    chunks_by_sub_query: dict[str, list[RetrievedChunk]]
    dense_candidates_total: int
    bm25_candidates_total: int

    # Điều khiển vòng lặp & chấm bằng chứng
    hop_count: int
    last_grade: EvidenceGrade | None
    grades: list[EvidenceGrade]
    used_fast_path: bool
    used_external_arxiv: bool
    external_ingested_docs: list[str]
    next_action: AgentNextAction

    # Quan sát thực thi & đầu ra cuối
    llm_calls: int
    steps: list[AgentStepTrace]
    grounded_answer: GroundedAnswer | None


def create_initial_state(
    query: str,
    topic_id: str | None = None,
    doc_ids: list[str] | None = None,
    max_hops: int = 2,
    top_k: int = 6,
    allow_external_arxiv: bool = False,
    chat_history: list[ChatTurn] | None = None,
) -> ResearchAgentState:
    return ResearchAgentState(
        query=query,
        topic_id=topic_id,
        doc_ids=doc_ids,
        max_hops=max(1, max_hops),
        top_k=max(1, top_k),
        allow_external_arxiv=allow_external_arxiv,
        chat_history=list(chat_history) if chat_history else [],
        plan=None,
        pending_queries=[],
        executed_queries=[],
        pre_rerank_chunks=[],
        evidence_chunks=[],
        chunks_by_sub_query={},
        dense_candidates_total=0,
        bm25_candidates_total=0,
        hop_count=0,
        last_grade=None,
        grades=[],
        used_fast_path=False,
        used_external_arxiv=False,
        external_ingested_docs=[],
        next_action="synthesize",
        llm_calls=0,
        steps=[],
        grounded_answer=None,
    )