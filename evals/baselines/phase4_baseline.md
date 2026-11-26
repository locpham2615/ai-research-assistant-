# RAG Evaluation Summary — `rag_golden_v1`

## 1. Ablation Comparison

| Mode | Samples | DocRec@Pre | DocRec@Post | EvHit@Post | MRR@Post | NDCG@Post | Grounded | Abstain F1 | False Refusal | Leak Rate | p50 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `dense_only` | 60 | 100.0% | 92.9% | 56.1% | 0.586 | 0.620 | 76.0% | 0.581 | 24.0% | 10.0% | 4163 ms |
| `hybrid_rrf` | 60 | 100.0% | 95.9% | 70.2% | 0.705 | 0.740 | 90.0% | 0.800 | 10.0% | 0.0% | 6436 ms |
| `full_rag` | 60 | 100.0% | 98.0% | 67.8% | 0.715 | 0.739 | 90.0% | 0.800 | 10.0% | 0.0% | 7222 ms |

## 3. Stage Latency Breakdown (Mean / p50 / p95)

| Mode | Retrieve | Rerank | Generate | Total |
| :--- | :---: | :---: | :---: | :---: |
| `dense_only` | 7 ms / 7 ms / 9 ms | 0 ms / 0 ms / 0 ms | 4435 ms / 4157 ms / 6029 ms | 4442 ms / 4163 ms / 6036 ms |
| `hybrid_rrf` | 2011 ms / 1980 ms / 2489 ms | 0 ms / 0 ms / 0 ms | 4533 ms / 4241 ms / 6543 ms | 6543 ms / 6436 ms / 8379 ms |
| `full_rag` | 2028 ms / 1987 ms / 2543 ms | 966 ms / 947 ms / 1449 ms | 4290 ms / 4173 ms / 5881 ms | 7284 ms / 7222 ms / 8715 ms |

## 4. Breakdown by Category

### Mode: `dense_only`

| Category | Count | Grounded Rate | DocRec@Post | EvHit@Post | MRR@Post |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `conceptual` | 15 | 93.3% | 100.0% | 58.9% | 0.630 |
| `cross_lingual_vi` | 12 | 58.3% | 91.7% | 47.9% | 0.458 |
| `factoid` | 15 | 80.0% | 100.0% | 63.3% | 0.572 |
| `multi_paper` | 8 | 62.5% | 68.8% | 50.0% | 0.729 |
| `unanswerable` | 10 | 10.0% | — | — | — |

### Mode: `hybrid_rrf`

| Category | Count | Grounded Rate | DocRec@Post | EvHit@Post | MRR@Post |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `conceptual` | 15 | 93.3% | 100.0% | 61.9% | 0.592 |
| `cross_lingual_vi` | 12 | 83.3% | 100.0% | 68.1% | 0.750 |
| `factoid` | 15 | 100.0% | 100.0% | 85.0% | 0.744 |
| `multi_paper` | 8 | 75.0% | 75.0% | 60.4% | 0.760 |
| `unanswerable` | 10 | 0.0% | — | — | — |

### Mode: `full_rag`

| Category | Count | Grounded Rate | DocRec@Post | EvHit@Post | MRR@Post |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `conceptual` | 15 | 100.0% | 100.0% | 48.9% | 0.583 |
| `cross_lingual_vi` | 12 | 91.7% | 100.0% | 81.9% | 0.771 |
| `factoid` | 15 | 93.3% | 100.0% | 85.0% | 0.833 |
| `multi_paper` | 8 | 62.5% | 87.5% | 50.0% | 0.656 |
| `unanswerable` | 10 | 0.0% | — | — | — |

## 5. Failure Analysis

### Mode: `dense_only` (15 issues)

| Sample | Category | Failure Type | DocRec Pre -> Post | Invalid Cites | Abstain Reason |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `q001` | `factoid` | `false_refusal_llm_or_citation` | 100.0% -> 100.0% | 0 | The sources do not specify the hardware setup, per-step duration, total training steps, or total training time for the base Transformer model. While the big model's training steps and duration are mentioned, these do not directly apply to the base model. The per-step duration and hardware details are missing. |
| `q012` | `factoid` | `false_refusal_llm_or_citation` | 100.0% -> 100.0% | 0 | The sources do not provide specific default values for ISREL, ISSUP, ISUSE, retrieval threshold, or segment-level beam width during inference. |
| `q015` | `factoid` | `false_refusal_llm_or_citation` | 100.0% -> 100.0% | 0 | The sources do not list the five external tools incorporated by Toolformer nor specify which models implement the Question Answering and Machine Translation tools. |
| `q020` | `conceptual` | `runtime_error:SchemaValidationFailedError` | — -> — | 0 | runtime_error:SchemaValidationFailedError |
| `q034` | `cross_lingual_vi` | `false_refusal_llm_or_citation` | 100.0% -> 100.0% | 0 | The sources do not provide information on how the trade-off between swapping and recomputation changes with block size for KV cache recovery in vLLM. |
| `q035` | `cross_lingual_vi` | `false_refusal_llm_or_citation` | 100.0% -> 100.0% | 0 | The sources do not provide information on which specific attention matrix weights to apply LoRA to for optimal accuracy in GPT-3 175B. |
| `q037` | `cross_lingual_vi` | `false_refusal_rerank_over_filter` | 100.0% -> 0.0% | 0 | Các nguồn không đề cập đến kỹ thuật 'index hot-swapping' hay tỷ lệ trả lời đúng được thu được từ việc sử dụng kỹ thuật này. |
| `q038` | `cross_lingual_vi` | `false_refusal_llm_or_citation` | 100.0% -> 100.0% | 0 | Các nguồn không đề cập đến khái niệm 'Retrieval Collapse' hoặc nguyên nhân gây ra nó. Chỉ có đề cập đến việc giảm tần suất truy vấn ảnh hưởng đến hiệu suất, nhưng không liên kết với hiện tượng 'Retrieval Collapse'. |
| `q042` | `cross_lingual_vi` | `false_refusal_llm_or_citation` | 100.0% -> 100.0% | 0 | Các nguồn không đề cập đến việc khả năng sử dụng công cụ ngoài bắt đầu xuất hiện ở ngưỡng kích thước nào của mô hình, cũng như không có thông tin về ngoại lệ liên quan đến số tham số. |
| `q045` | `multi_paper` | `partial_doc_recall` | 100.0% -> 50.0% | 0 | — |
| `q046` | `multi_paper` | `false_refusal_llm_or_citation` | 100.0% -> 50.0% | 0 | The sources do not mention ReAct or provide a comparison between ReAct and Toolformer regarding how they teach language models to use external tools or support interactive tool use. |
| `q047` | `multi_paper` | `false_refusal_llm_or_citation` | 100.0% -> 50.0% | 0 | The sources do not contain information about GraphRAG or its application of self-reflection, making a comparative analysis impossible. |
| `q048` | `multi_paper` | `partial_doc_recall` | 100.0% -> 50.0% | 0 | — |
| `q050` | `multi_paper` | `false_refusal_llm_or_citation` | 100.0% -> 50.0% | 0 | Các nguồn không cung cấp thông tin cụ thể về giai đoạn sử dụng vLLM trong Self-RAG và không mô tả chi tiết cách cơ chế chia sẻ block hỗ trợ hiệu quả giải mã beam search. |
| `q060` | `unanswerable` | `unanswerable_leak` | — -> — | 0 | — |

### Mode: `hybrid_rrf` (7 issues)

| Sample | Category | Failure Type | DocRec Pre -> Post | Invalid Cites | Abstain Reason |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `q020` | `conceptual` | `runtime_error:SchemaValidationFailedError` | — -> — | 0 | runtime_error:SchemaValidationFailedError |
| `q038` | `cross_lingual_vi` | `false_refusal_llm_or_citation` | 100.0% -> 100.0% | 0 | Các nguồn tham khảo không đề cập đến hiện tượng 'Retrieval Collapse' hay nguyên nhân gây ra nó. |
| `q042` | `cross_lingual_vi` | `false_refusal_llm_or_citation` | 100.0% -> 100.0% | 0 | The sources do not provide information on the specific parameter threshold for the emergence of tool usage capabilities or any exceptions to it. |
| `q043` | `multi_paper` | `partial_doc_recall` | 100.0% -> 50.0% | 0 | — |
| `q044` | `multi_paper` | `false_refusal_llm_or_citation` | 100.0% -> 50.0% | 0 | The QLoRA paper is not mentioned or referenced in the provided sources, and thus no information is available to compare its findings with those of the original LoRA paper on which Transformer weight matrices should receive LoRA adapters. |
| `q047` | `multi_paper` | `false_refusal_llm_or_citation` | 100.0% -> 50.0% | 0 | The sources do not mention GraphRAG or its use of self-reflection, making a comparative analysis impossible. |
| `q048` | `multi_paper` | `partial_doc_recall` | 100.0% -> 50.0% | 0 | — |

### Mode: `full_rag` (6 issues)

| Sample | Category | Failure Type | DocRec Pre -> Post | Invalid Cites | Abstain Reason |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `q014` | `factoid` | `false_refusal_llm_or_citation` | 100.0% -> 100.0% | 0 | The sources do not explicitly define or list three action types within the Wikipedia web API action space used by ReAct on HotpotQA and FEVER. |
| `q038` | `cross_lingual_vi` | `false_refusal_llm_or_citation` | 100.0% -> 100.0% | 0 | Các nguồn tham khảo không đề cập đến hiện tượng 'Retrieval Collapse' hoặc các nguyên nhân gây ra nó. |
| `q045` | `multi_paper` | `false_refusal_llm_or_citation` | 100.0% -> 100.0% | 0 | The sources lack a direct comparison of Self-RAG's retrieval mechanism with the latent-document retrieval approach in the original RAG paper, such as the RAG-Sequence or RAG-Token models, and do not detail how Self-RAG's retrieval differs in structure or function from those methods. |
| `q047` | `multi_paper` | `false_refusal_llm_or_citation` | 100.0% -> 50.0% | 0 | The sources do not provide information about GraphRAG or its self-reflection mechanism, making a comparative analysis impossible. |
| `q048` | `multi_paper` | `partial_doc_recall` | 100.0% -> 50.0% | 0 | — |
| `q050` | `multi_paper` | `false_refusal_llm_or_citation` | 100.0% -> 100.0% | 0 | Các nguồn không nêu rõ giai đoạn sử dụng vLLM trong Self-RAG và không cung cấp chi tiết về cách cơ chế chia sẻ block hỗ trợ hiệu quả giải mã beam search. |
