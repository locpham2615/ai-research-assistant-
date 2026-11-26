import json
import re
from collections import Counter
from pathlib import Path

from ai_research_assistant.eval.schemas import DatasetAuditReport, EvalDataset

_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)


def normalize_doc_id(doc_id: str) -> str:
    """Chuẩn hóa doc_id bằng cách bỏ hậu tố phiên bản arXiv (vd: arxiv_1706.03762v7)."""
    return _ARXIV_VERSION_RE.sub("", doc_id.strip())


def load_eval_dataset(path: Path) -> EvalDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return EvalDataset.model_validate(raw)


def audit_eval_dataset(dataset: EvalDataset) -> DatasetAuditReport:
    errors: list[str] = []
    warnings: list[str] = []

    corpus_ids = {normalize_doc_id(p.doc_id) for p in dataset.corpus}
    if not corpus_ids:
        errors.append("Dataset corpus is empty.")

    if not dataset.samples:
        errors.append("Dataset has no samples.")

    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    category_counts: Counter[str] = Counter()
    doc_coverage: Counter[str] = Counter({doc_id: 0 for doc_id in corpus_ids})

    for idx, sample in enumerate(dataset.samples, start=1):
        prefix = f"[{sample.sample_id or f'index_{idx}'}]"
        category_counts[sample.category] += 1

        if not sample.sample_id.strip():
            errors.append(f"{prefix} Empty sample_id.")
        elif sample.sample_id in seen_ids:
            errors.append(f"{prefix} Duplicate sample_id.")
        seen_ids.add(sample.sample_id)

        norm_q = re.sub(r"\s+", " ", sample.question.strip().lower())
        if not norm_q:
            errors.append(f"{prefix} Empty question.")
        elif norm_q in seen_questions:
            errors.append(f"{prefix} Duplicate question.")
        seen_questions.add(norm_q)

        if not sample.reference_answer.strip():
            errors.append(f"{prefix} Empty reference_answer.")

        if sample.topic_id != dataset.topic_id:
            warnings.append(
                f"{prefix} topic_id='{sample.topic_id}' differs from "
                f"dataset topic_id='{dataset.topic_id}'."
            )

        norm_expected_docs = [normalize_doc_id(d) for d in sample.expected_doc_ids]

        # Kiểm tra tính nhất quán giữa category='unanswerable' và should_abstain
        if sample.category == "unanswerable" and not sample.should_abstain:
            errors.append(f"{prefix} category='unanswerable' must set should_abstain=True.")
        if sample.should_abstain and sample.category != "unanswerable":
            warnings.append(
                f"{prefix} should_abstain=True with category='{sample.category}'."
            )

        if sample.should_abstain:
            if norm_expected_docs:
                errors.append(f"{prefix} Abstain sample must have empty expected_doc_ids.")
            if sample.expected_evidence_keywords:
                errors.append(
                    f"{prefix} Abstain sample must have empty expected_evidence_keywords."
                )
            if sample.evidence_quote.strip():
                warnings.append(f"{prefix} Abstain sample should have empty evidence_quote.")
            continue

        # Nhánh câu hỏi có thể trả lời (should_abstain == False)
        if not norm_expected_docs:
            errors.append(f"{prefix} Answerable sample missing expected_doc_ids.")
        for doc_id in norm_expected_docs:
            if doc_id not in corpus_ids:
                errors.append(f"{prefix} expected_doc_id '{doc_id}' not found in corpus.")
            else:
                doc_coverage[doc_id] += 1

        if sample.category == "multi_paper" and len(set(norm_expected_docs)) < 2:
            errors.append(
                f"{prefix} multi_paper sample must reference at least 2 distinct doc_ids."
            )

        quote_lower = sample.evidence_quote.strip().lower()
        if not quote_lower:
            errors.append(f"{prefix} Answerable sample missing evidence_quote.")

        if not sample.expected_evidence_keywords:
            errors.append(f"{prefix} Answerable sample missing expected_evidence_keywords.")
        else:
            if len(sample.expected_evidence_keywords) < 2:
                warnings.append(
                    f"{prefix} Fewer than 2 expected_evidence_keywords "
                    f"({sample.expected_evidence_keywords})."
                )
            for kw in sample.expected_evidence_keywords:
                kw_clean = kw.strip().lower()
                if not kw_clean:
                    errors.append(f"{prefix} Contains blank keyword.")
                elif quote_lower and kw_clean not in quote_lower:
                    errors.append(
                        f"{prefix} Keyword '{kw}' not found verbatim in evidence_quote."
                    )

    for doc_id, count in sorted(doc_coverage.items()):
        if count == 0:
            warnings.append(f"Corpus paper '{doc_id}' has 0 samples covering it.")

    return DatasetAuditReport(
        dataset_name=dataset.dataset_name,
        total_papers=len(dataset.corpus),
        total_samples=len(dataset.samples),
        category_counts=dict(sorted(category_counts.items())),
        doc_coverage=dict(sorted(doc_coverage.items())),
        errors=errors,
        warnings=warnings,
    )