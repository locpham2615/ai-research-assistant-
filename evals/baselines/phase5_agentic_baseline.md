# RAG Evaluation Summary — `rag_golden_v1`

## 1. Ablation Comparison

| Mode | Samples | DocRec@Pre | DocRec@Post | EvHit@Post | MRR@Post | NDCG@Post | Grounded | Abstain F1 | False Refusal | Leak Rate | p50 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `agentic_rag` | 60 | 99.0% | 96.9% | 70.1% | 0.713 | 0.729 | 86.0% | 0.692 | 14.0% | 10.0% | 12215 ms |

## Agentic Workflow Summary

| Mode | Simple | Decompose | Fast-Path | Multi-Hop (>1) | Avg Hops | Avg LLM Calls |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `agentic_rag` | 37 | 22 | 35 | 7 | 1.12 | 2.34 |

## 3. Stage Latency Breakdown (Mean / p50 / p95)

| Mode | Retrieve | Rerank | Generate | Total |
| :--- | :---: | :---: | :---: | :---: |
| `agentic_rag` | 6199 ms / 4889 ms / 10313 ms | 2203 ms / 0 ms / 9226 ms | 4648 ms / 4716 ms / 7120 ms | 13365 ms / 12215 ms / 22702 ms |

## 4. Breakdown by Category

### Mode: `agentic_rag`

| Category | Count | Grounded Rate | DocRec@Post | EvHit@Post | MRR@Post |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `conceptual` | 15 | 86.7% | 100.0% | 62.5% | 0.568 |
| `cross_lingual_vi` | 12 | 83.3% | 91.7% | 71.5% | 0.771 |
| `factoid` | 15 | 86.7% | 100.0% | 85.0% | 0.878 |
| `multi_paper` | 8 | 87.5% | 93.8% | 53.1% | 0.573 |
| `unanswerable` | 10 | 10.0% | — | — | — |

## 5. Failure Analysis

### Mode: `agentic_rag` (9 issues)

| Sample | Category | Failure Type | DocRec Pre -> Post | Invalid Cites | Abstain Reason |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `q011` | `factoid` | `false_refusal_llm_or_citation` | 100.0% -> 100.0% | 2 | no_valid_citations |
| `q014` | `factoid` | `false_refusal_llm_or_citation` | 100.0% -> 100.0% | 0 | The sources do not explicitly mention or define the three action types in the Wikipedia web API action space used by ReAct on HotpotQA and FEVER. The available information discusses reasoning processes and interactions with Wikipedia, but lacks a clear enumeration of the action types. |
| `q022` | `conceptual` | `false_refusal_llm_or_citation` | 100.0% -> 100.0% | 0 | The sources do not contain information about the training recipe of RAG or the rationale for freezing the DPR document encoder, making it impossible to determine why it is frozen instead of being fine-tuned end-to-end. |
| `q029` | `conceptual` | `runtime_error:SchemaValidationFailedError` | — -> — | 0 | runtime_error:SchemaValidationFailedError |
| `q038` | `cross_lingual_vi` | `false_refusal_llm_or_citation` | 100.0% -> 100.0% | 0 | Các nguồn tham khảo không đề cập đến hiện tượng 'Retrieval Collapse' hay bất kỳ thông tin nào liên quan đến nó trong bối cảnh RAG gốc hoặc các phương pháp tự phản biện như SELF-RAG. |
| `q042` | `cross_lingual_vi` | `false_refusal_rerank_over_filter` | 100.0% -> 0.0% | 0 | no_relevant_evidence |
| `q048` | `multi_paper` | `partial_doc_recall` | 50.0% -> 50.0% | 0 | — |
| `q050` | `multi_paper` | `false_refusal_llm_or_citation` | 100.0% -> 100.0% | 0 | Không có thông tin trong các nguồn về việc Self-RAG sử dụng vLLM trong giai đoạn nào của thực nghiệm, cũng như không có mô tả về cách cơ chế chia sẻ block của vLLM hỗ trợ giải mã beam search trong Self-RAG. Các nguồn chỉ đề cập đến chức năng chia sẻ block của vLLM trong beam search nhưng không liên hệ với Self-RAG hoặc quá trình giải mã của nó. |
| `q060` | `unanswerable` | `unanswerable_leak` | — -> — | 0 | — |
