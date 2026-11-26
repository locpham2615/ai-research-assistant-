from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ai_research_assistant.retrieval.schemas import GroundedAnswer, RetrievedChunk

AgentRoute = Literal["simple", "decompose"]
AgentNextAction = Literal["synthesize", "refine", "arxiv_discovery", "abstain"]


class SubQuery(BaseModel):
    sub_query_id: str = Field(default="sq1")
    search_query: str = Field(min_length=1)
    bm25_keywords: list[str] = Field(default_factory=list)
    target_aspect: str = Field(default="")

    @field_validator("search_query", "target_aspect", "sub_query_id", mode="before")
    @classmethod
    def _strip_text(cls, value: object) -> str:
        return str(value).strip() if value is not None else ""

    @field_validator("bm25_keywords", mode="before")
    @classmethod
    def _clean_keywords(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned[:6]


class QueryPlan(BaseModel):
    route: AgentRoute = "simple"
    sub_queries: list[SubQuery] = Field(min_length=1, max_length=3)
    reasoning: str = Field(default="")

    @field_validator("sub_queries", mode="before")
    @classmethod
    def _cap_sub_queries(cls, value: object) -> object:
        if isinstance(value, list) and len(value) > 3:
            return value[:3]
        return value

    @model_validator(mode="after")
    def _normalize_plan(self) -> "QueryPlan":
        self.route = "decompose" if len(self.sub_queries) > 1 else "simple"
        seen_ids: set[str] = set()
        for idx, sq in enumerate(self.sub_queries, start=1):
            if not sq.sub_query_id or sq.sub_query_id in seen_ids:
                sq.sub_query_id = f"sq{idx}"
            seen_ids.add(sq.sub_query_id)
        return self


class EvidenceGrade(BaseModel):
    is_sufficient: bool
    missing_aspects: list[str] = Field(default_factory=list)
    follow_up_queries: list[SubQuery] = Field(default_factory=list, max_length=2)
    rationale: str = Field(default="")

    @field_validator("missing_aspects", mode="before")
    @classmethod
    def _clean_aspects(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("follow_up_queries", mode="before")
    @classmethod
    def _cap_follow_ups(cls, value: object) -> object:
        if isinstance(value, list) and len(value) > 2:
            return value[:2]
        return value


class AgentStepTrace(BaseModel):
    node: str
    hop: int = 0
    latency_ms: float = 0.0
    detail: str = ""


class AgentExecutionTrace(BaseModel):
    route: AgentRoute = "simple"
    hops_executed: int = 0
    used_fast_path: bool = False
    used_external_arxiv: bool = False
    llm_calls: int = 0
    plan: QueryPlan | None = None
    grades: list[EvidenceGrade] = Field(default_factory=list)
    steps: list[AgentStepTrace] = Field(default_factory=list)

class AgenticAnswer(BaseModel):
    grounded_answer: GroundedAnswer
    agent_trace: AgentExecutionTrace = Field(default_factory=AgentExecutionTrace)
    pre_rerank_chunks: list[RetrievedChunk] = Field(default_factory=list)