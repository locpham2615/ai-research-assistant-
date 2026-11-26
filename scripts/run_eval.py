import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from ai_research_assistant.config import get_settings
from ai_research_assistant.db.sqlite import SQLiteStore
from ai_research_assistant.db.vector_store import VectorStore
from ai_research_assistant.eval.dataset import audit_eval_dataset, load_eval_dataset
from ai_research_assistant.eval.judge import RAGJudge
from ai_research_assistant.eval.reporter import (
    classify_failure,
    save_eval_report_json,
    save_markdown_report,
)
from ai_research_assistant.eval.runner import EvalMode, EvalRunner
from ai_research_assistant.eval.schemas import EvalDataset, EvalReport, SampleEvalResult
from ai_research_assistant.harness.llm_router import LLMRouter
from ai_research_assistant.harness.schema_guard import SchemaGuard
from ai_research_assistant.pipelines.chunkers.section_chunker import SectionChunker
from ai_research_assistant.pipelines.embedders.text_embedder import TextEmbedder
from ai_research_assistant.pipelines.ingest_pipeline import IngestPipeline
from ai_research_assistant.pipelines.parsers.document_parser import DocumentParser

ALL_MODES: tuple[EvalMode, ...] = ("dense_only", "hybrid_rrf", "full_rag")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation and ablation benchmarks on a Golden Dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evals/datasets/rag_golden_v1.json"),
        help="Path to the golden evaluation dataset JSON.",
    )
    parser.add_argument(
        "--mode",
        choices=["dense_only", "hybrid_rrf", "full_rag", "all"],
        default="full_rag",
        help="Ablation mode to execute.",
    )
    parser.add_argument(
        "--ingest-corpus",
        action="store_true",
        help="Ingest all papers listed in dataset.corpus into dataset.topic_id first.",
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Force re-ingestion even if documents are already indexed.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/eval_pdfs"),
        help="Directory storing downloaded arXiv PDFs.",
    )
    parser.add_argument(
        "--with-judge",
        action="store_true",
        help="Enable LLM-as-a-Judge scoring (faithfulness, correctness, relevancy).",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="",
        help="Optional comma-separated category filter (e.g. factoid,cross_lingual_vi).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of samples to run (for quick smoke checks).",
    )
    parser.add_argument(
        "--rerank-min-score",
        type=float,
        default=None,
        help="Optional override for CrossEncoderReranker min_score threshold.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to store JSON and Markdown evaluation reports.",
    )
    return parser.parse_args()


def _resolve_modes(mode_arg: str) -> list[EvalMode]:
    if mode_arg == "all":
        return list(ALL_MODES)
    if mode_arg == "dense_only":
        return ["dense_only"]
    if mode_arg == "hybrid_rrf":
        return ["hybrid_rrf"]
    return ["full_rag"]


def _format_progress(idx: int, total: int, result: SampleEvalResult) -> str:
    failure = classify_failure(result)
    status = failure if failure is not None else "ok"
    rec = (
        f"{result.doc_recall_post_rerank:.2f}"
        if result.doc_recall_post_rerank is not None
        else "n/a"
    )
    mrr = f"{result.mrr_post_rerank:.2f}" if result.mrr_post_rerank is not None else "n/a"
    return (
        f"  [{idx:02d}/{total:02d}] {result.sample_id} ({result.category:<16}) | "
        f"grounded={str(result.is_grounded):<5} | rec={rec} | mrr={mrr} | "
        f"{result.latency.total_ms:6.0f}ms | {status}"
    )


async def ingest_corpus(
    dataset: EvalDataset,
    pipeline: IngestPipeline,
    pdf_dir: Path,
    force: bool,
) -> tuple[int, int]:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    ok_count = 0
    fail_count = 0

    print(
        f"\n=== INGESTING CORPUS ({len(dataset.corpus)} papers -> "
        f"topic='{dataset.topic_id}', force={force}) ==="
    )
    for idx, paper in enumerate(dataset.corpus, start=1):
        res = await pipeline.ingest_arxiv(
            paper.arxiv_id,
            topic_id=dataset.topic_id,
            download_dir=pdf_dir,
            force=force,
        )
        print(
            f"  [{idx:02d}/{len(dataset.corpus):02d}] {paper.arxiv_id} -> "
            f"status={res.status} | chunks={res.chunks_count} | "
            f"note={res.skipped_reason or res.error_message or 'ok'}"
        )
        if res.status == "failed":
            fail_count += 1
        else:
            ok_count += 1

    return ok_count, fail_count


async def run_async(args: argparse.Namespace) -> int:
    if not args.dataset.exists():
        print(f"ERROR: Dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    dataset = load_eval_dataset(args.dataset)
    audit = audit_eval_dataset(dataset)
    if not audit.is_valid:
        print("ERROR: Dataset audit failed:", file=sys.stderr)
        for err in audit.errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    settings = get_settings()
    db = SQLiteStore(settings.sqlite_db_path)
    await db.init_db()

    vs = VectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection_name,
    )
    vs.ensure_collection(vector_size=384)

    router = LLMRouter(settings=settings)
    guard = SchemaGuard(router)
    embedder = TextEmbedder(model_name=settings.embedding_model_name, sqlite_store=db)

    if args.ingest_corpus:
        pipeline = IngestPipeline(
            sqlite_store=db,
            vector_store=vs,
            parser=DocumentParser(grobid_url=settings.grobid_url),
            chunker=SectionChunker(),
            embedder=embedder,
            schema_guard=guard,
        )
        ok_count, fail_count = await ingest_corpus(
            dataset=dataset,
            pipeline=pipeline,
            pdf_dir=args.pdf_dir,
            force=args.force_ingest,
        )
        print(f"Corpus ingest complete: ok={ok_count}, failed={fail_count}")
        if fail_count > 0:
            print(
                "ERROR: Aborting benchmark because one or more corpus papers failed to ingest.",
                file=sys.stderr,
            )
            return 1

    categories: set[str] | None = None
    if args.categories.strip():
        categories = {c.strip() for c in args.categories.split(",") if c.strip()}

    judge: RAGJudge | None = None
    if args.with_judge:
        judge = RAGJudge(
            schema_guard=guard,
            max_evidence_chars=settings.eval_judge_max_evidence_chars,
        )

    min_score = (
        args.rerank_min_score
        if args.rerank_min_score is not None
        else settings.rerank_min_score
    )
    modes = _resolve_modes(args.mode)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    reports: list[EvalReport] = []

    for mode in modes:
        print(
            f"\n=== RUNNING MODE: {mode} "
            f"(judge={'on' if judge else 'off'}, min_score={min_score}) ==="
        )
        runner = EvalRunner.from_components(
            mode=mode,
            embedder=embedder,
            vector_store=vs,
            sqlite_store=db,
            schema_guard=guard,
            reranker_model_name=settings.reranker_model_name,
            retrieval_top_k=settings.retrieval_top_k,
            rerank_top_k=settings.rerank_top_k,
            rerank_min_score=min_score,
            rrf_k=settings.rrf_k,
            max_context_chars=settings.max_context_chars,
            judge=judge,
        )
        report = await runner.run(
            dataset=dataset,
            categories=categories,
            limit=args.limit,
            on_sample_complete=lambda i, n, r: print(_format_progress(i, n, r)),
        )
        reports.append(report)

        json_path = args.output_dir / f"eval_{mode}_{timestamp}.json"
        save_eval_report_json(report, json_path)
        print(
            f"Saved JSON report: {json_path} | "
            f"DocRec@Post={report.retrieval.doc_recall_post_rerank * 100:.1f}% | "
            f"MRR={report.retrieval.mrr_post_rerank:.3f} | "
            f"AbstainF1={report.abstention.f1:.3f} | "
            f"p50={report.latency.total.p50_ms:.0f}ms"
        )

    md_ts_path = args.output_dir / f"eval_summary_{timestamp}.md"
    md_latest_path = args.output_dir / "eval_summary_latest.md"
    save_markdown_report(reports, md_ts_path)
    save_markdown_report(reports, md_latest_path)
    print(f"\nSaved Markdown summary: {md_ts_path}")
    print(f"Updated latest summary: {md_latest_path}")
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run_async(args))


if __name__ == "__main__":
    raise SystemExit(main())