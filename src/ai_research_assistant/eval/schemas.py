from typing import Literal

from pydantic import BaseModel, Field, field_validator

EvalCategory = Literal[
    "factoid",
    "conceptual",
    "cross_lingual_vi",
    "multi_paper",
    "unanswerable",
]
EvalDifficulty = Literal["easy", "medium", "hard"]


class CorpusPaper(BaseModel):
    doc_id: str
    arxiv_id: str
    title: str


class EvalSample(BaseModel):
    sample_id: str
    question: str
    topic_id: str
    category: EvalCategory
    difficulty: EvalDifficulty = "medium"
    should_abstain: bool
    expected_doc_ids: list[str] = Field(default_factory=list)
    expected_sections: list[str] = Field(default_factory=list)
    expected_evidence_keywords: list[str] = Field(default_factory=list)
    reference_answer: str
    evidence_quote: str = ""

    @field_validator("difficulty", mode="before")
    @classmethod
    def _normalize_difficulty(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class EvalDataset(BaseModel):
    dataset_name: str
    version: str = "1.0"
    topic_id: str
    corpus: list[CorpusPaper] = Field(default_factory=list)
    samples: list[EvalSample] = Field(default_factory=list)


class DatasetAuditReport(BaseModel):
    dataset_name: str
    total_papers: int
    total_samples: int
    category_counts: dict[str, int] = Field(default_factory=dict)
    doc_coverage: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


class StageLatency(BaseModel):
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0


class JudgeScores(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    correctness: float = Field(ge=0.0, le=1.0)
    relevancy: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class SampleEvalResult(BaseModel):
    sample_id: str
    category: EvalCategory
    should_abstain: bool
    is_grounded: bool
    abstain_reason: str = ""
    answer: str = ""
    doc_recall_pre_rerank: float | None = None
    doc_recall_post_rerank: float | None = None
    evidence_hit_pre_rerank: float | None = None
    evidence_hit_post_rerank: float | None = None
    mrr_post_rerank: float | None = None
    ndcg_post_rerank: float | None = None
    has_valid_citation: bool = False
    invalid_citations_count: int = 0
    latency: StageLatency = Field(default_factory=StageLatency)
    judge: JudgeScores | None = None