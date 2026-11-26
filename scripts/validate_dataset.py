import json
import sys
from pathlib import Path

from pydantic import ValidationError

from ai_research_assistant.eval.dataset import audit_eval_dataset, load_eval_dataset


def main() -> int:
    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("evals/datasets/rag_golden_v1.json")
    )
    if not target.exists():
        print(f"ERROR: File not found: {target}")
        return 1

    try:
        dataset = load_eval_dataset(target)
    except json.JSONDecodeError as exc:
        print(f"JSON SYNTAX ERROR at line {exc.lineno}, col {exc.colno}: {exc.msg}")
        return 1
    except ValidationError as exc:
        print("SCHEMA VALIDATION ERROR:")
        print(exc)
        return 1

    report = audit_eval_dataset(dataset)

    print(f"=== DATASET AUDIT: {report.dataset_name} ({target}) ===")
    print(f"Papers in corpus : {report.total_papers}")
    print(f"Total samples    : {report.total_samples}")

    print("\nCategory distribution:")
    for cat, count in report.category_counts.items():
        print(f"  - {cat:<18}: {count}")

    print("\nPaper coverage (number of samples referencing each doc_id):")
    for doc_id, count in report.doc_coverage.items():
        print(f"  - {doc_id:<18}: {count}")

    if report.warnings:
        print(f"\nWARNINGS ({len(report.warnings)}):")
        for msg in report.warnings:
            print(f"  [WARN] {msg}")

    if report.errors:
        print(f"\nERRORS ({len(report.errors)}):")
        for msg in report.errors:
            print(f"  [ERR]  {msg}")
        print("\nRESULT: FAILED")
        return 1

    print("\nRESULT: PASSED (0 errors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())