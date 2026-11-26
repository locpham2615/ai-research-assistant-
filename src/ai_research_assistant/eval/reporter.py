from pathlib import Path

from ai_research_assistant.eval.schemas import EvalReport, SampleEvalResult


def classify_failure(result: SampleEvalResult) -> str | None:
    """Phân loại lỗi cốt lõi của một mẫu để phục vụ debug nhanh."""
    if result.abstain_reason.startswith("runtime_error:"):
        return result.abstain_reason

    if result.should_abstain:
        if result.is_grounded:
            return "unanswerable_leak"
        return None

    if not result.is_grounded:
        if (result.doc_recall_pre_rerank or 0.0) == 0.0:
            return "false_refusal_retrieval_miss"
        if (result.doc_recall_post_rerank or 0.0) == 0.0:
            return "false_refusal_rerank_over_filter"
        return "false_refusal_llm_or_citation"

    if result.invalid_citations_count > 0:
        return "hallucinated_citation"

    if (result.doc_recall_post_rerank or 0.0) < 1.0:
        return "partial_doc_recall"

    if result.judge is not None and (
        result.judge.faithfulness < 0.5 or result.judge.correctness < 0.5
    ):
        return "low_judge_score"

    return None


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _fmt_score(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _fmt_ms(value: float) -> str:
    return f"{value:.0f} ms"


def save_eval_report_json(report: EvalReport, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return output_path


def render_markdown_report(
    reports: list[EvalReport],
    max_failures_per_report: int = 15,
) -> str:
    if not reports:
        return "# RAG Evaluation Report\n\nNo evaluation reports provided.\n"

    dataset_name = reports[0].dataset_name
    lines: list[str] = [
        f"# RAG Evaluation Summary — `{dataset_name}`",
        "",
        "## 1. Ablation Comparison",
        "",
        (
            "| Mode | Samples | DocRec@Pre | DocRec@Post | EvHit@Post | MRR@Post | "
            "NDCG@Post | Grounded | Abstain F1 | False Refusal | Leak Rate | p50 Latency |"
        ),
        (
            "| :--- | :---: | :---: | :---: | :---: | :---: | "
            ":---: | :---: | :---: | :---: | :---: | :---: |"
        ),
    ]

    for rep in reports:
        lines.append(
            f"| `{rep.mode}` | {rep.total_samples} "
            f"| {_fmt_pct(rep.retrieval.doc_recall_pre_rerank)} "
            f"| {_fmt_pct(rep.retrieval.doc_recall_post_rerank)} "
            f"| {_fmt_pct(rep.retrieval.evidence_hit_post_rerank)} "
            f"| {_fmt_score(rep.retrieval.mrr_post_rerank)} "
            f"| {_fmt_score(rep.retrieval.ndcg_post_rerank)} "
            f"| {_fmt_pct(rep.citation.grounded_answer_rate)} "
            f"| {_fmt_score(rep.abstention.f1)} "
            f"| {_fmt_pct(rep.abstention.false_refusal_rate)} "
            f"| {_fmt_pct(rep.abstention.unanswerable_leak_rate)} "
            f"| {_fmt_ms(rep.latency.total.p50_ms)} |"
        )

    judged_reports = [r for r in reports if r.judge is not None]
    if judged_reports:
        lines.extend(
            [
                "",
                "## 2. LLM-as-a-Judge Scores",
                "",
                "| Mode | Evaluated | Faithfulness | Correctness | Relevancy |",
                "| :--- | :---: | :---: | :---: | :---: |",
            ]
        )
        for rep in judged_reports:
            assert rep.judge is not None
            lines.append(
                f"| `{rep.mode}` | {rep.judge.evaluated_count} "
                f"| {_fmt_score(rep.judge.mean_faithfulness)} "
                f"| {_fmt_score(rep.judge.mean_correctness)} "
                f"| {_fmt_score(rep.judge.mean_relevancy)} |"
            )

    agent_reports = [r for r in reports if r.agent is not None]
    if agent_reports:
        lines.extend(
            [
                "",
                "## Agentic Workflow Summary",
                "",
                (
                    "| Mode | Simple | Decompose | Fast-Path | Multi-Hop (>1) | "
                    "Avg Hops | Avg LLM Calls |"
                ),
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for rep in agent_reports:
            assert rep.agent is not None
            ag = rep.agent
            lines.append(
                f"| `{rep.mode}` | {ag.simple_count} | {ag.decompose_count} "
                f"| {ag.fast_path_count} | {ag.multi_hop_count} "
                f"| {ag.avg_hops:.2f} | {ag.avg_llm_calls:.2f} |"
            )

    lines.extend(
        [
            "",
            "## 3. Stage Latency Breakdown (Mean / p50 / p95)",
            "",
            "| Mode | Retrieve | Rerank | Generate | Total |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
    )
    for rep in reports:
        lat = rep.latency
        lines.append(
            f"| `{rep.mode}` "
            f"| {_fmt_ms(lat.retrieve.mean_ms)} / {_fmt_ms(lat.retrieve.p50_ms)} "
            f"/ {_fmt_ms(lat.retrieve.p95_ms)} "
            f"| {_fmt_ms(lat.rerank.mean_ms)} / {_fmt_ms(lat.rerank.p50_ms)} "
            f"/ {_fmt_ms(lat.rerank.p95_ms)} "
            f"| {_fmt_ms(lat.generate.mean_ms)} / {_fmt_ms(lat.generate.p50_ms)} "
            f"/ {_fmt_ms(lat.generate.p95_ms)} "
            f"| {_fmt_ms(lat.total.mean_ms)} / {_fmt_ms(lat.total.p50_ms)} "
            f"/ {_fmt_ms(lat.total.p95_ms)} |"
        )

    lines.extend(["", "## 4. Breakdown by Category", ""])
    for rep in reports:
        lines.extend(
            [
                f"### Mode: `{rep.mode}`",
                "",
                "| Category | Count | Grounded Rate | DocRec@Post | EvHit@Post | MRR@Post |",
                "| :--- | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for cat, summary in rep.by_category.items():
            lines.append(
                f"| `{cat}` | {summary.count} "
                f"| {_fmt_pct(summary.grounded_rate)} "
                f"| {_fmt_pct(summary.doc_recall_post_rerank)} "
                f"| {_fmt_pct(summary.evidence_hit_post_rerank)} "
                f"| {_fmt_score(summary.mrr_post_rerank)} |"
            )
        lines.append("")

    lines.extend(["## 5. Failure Analysis", ""])
    for rep in reports:
        failures = [
            (res, failure)
            for res in rep.results
            if (failure := classify_failure(res)) is not None
        ]
        lines.append(f"### Mode: `{rep.mode}` ({len(failures)} issues)")
        lines.append("")
        if not failures:
            lines.append("No failures detected.")
            lines.append("")
            continue

        lines.extend(
            [
                (
                    "| Sample | Category | Failure Type | DocRec Pre -> Post | "
                    "Invalid Cites | Abstain Reason |"
                ),
                "| :--- | :--- | :--- | :---: | :---: | :--- |",
            ]
        )
        for res, failure in failures[:max_failures_per_report]:
            rec_pair = (
                f"{_fmt_pct(res.doc_recall_pre_rerank)} -> "
                f"{_fmt_pct(res.doc_recall_post_rerank)}"
            )
            lines.append(
                f"| `{res.sample_id}` | `{res.category}` | `{failure}` "
                f"| {rec_pair} | {res.invalid_citations_count} "
                f"| {res.abstain_reason or '—'} |"
            )
        if len(failures) > max_failures_per_report:
            remaining = len(failures) - max_failures_per_report
            lines.append(f"\n*...and {remaining} more issues in JSON report.*")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def save_markdown_report(
    reports: list[EvalReport],
    output_path: Path,
    max_failures_per_report: int = 15,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = render_markdown_report(
        reports=reports,
        max_failures_per_report=max_failures_per_report,
    )
    output_path.write_text(content, encoding="utf-8")
    return output_path