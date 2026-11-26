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

class RetrievalMetrics(BaseModel):
    doc_recall_pre_rerank: float = 0.0
    doc_recall_post_rerank: float = 0.0
    evidence_hit_pre_rerank: float = 0.0
    evidence_hit_post_rerank: float = 0.0
    mrr_post_rerank: float = 0.0
    ndcg_post_rerank: float = 0.0


class CitationMetrics(BaseModel):
    grounded_answer_rate: float = 0.0
    valid_citation_rate: float = 0.0
    hallucinated_citation_rate: float = 0.0
    total_invalid_citations: int = 0


class AbstentionMetrics(BaseModel):
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    false_refusal_rate: float = 0.0
    unanswerable_leak_rate: float = 0.0


class LatencyPercentile(BaseModel):
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0


class LatencySummary(BaseModel):
    retrieve: LatencyPercentile = Field(default_factory=LatencyPercentile)
    rerank: LatencyPercentile = Field(default_factory=LatencyPercentile)
    generate: LatencyPercentile = Field(default_factory=LatencyPercentile)
    total: LatencyPercentile = Field(default_factory=LatencyPercentile)


class JudgeSummary(BaseModel):
    evaluated_count: int = 0
    mean_faithfulness: float = 0.0
    mean_correctness: float = 0.0
    mean_relevancy: float = 0.0


class CategorySummary(BaseModel):
    count: int = 0
    grounded_rate: float = 0.0
    doc_recall_post_rerank: float | None = None
    evidence_hit_post_rerank: float | None = None
    mrr_post_rerank: float | None = None


class EvalReport(BaseModel):
    dataset_name: str
    mode: str
    created_at: str
    total_samples: int
    answerable_samples: int
    unanswerable_samples: int
    retrieval: RetrievalMetrics
    citation: CitationMetrics
    abstention: AbstentionMetrics
    latency: LatencySummary
    judge: JudgeSummary | None = None
    by_category: dict[str, CategorySummary] = Field(default_factory=dict)
    results: list[SampleEvalResult] = Field(default_factory=list)