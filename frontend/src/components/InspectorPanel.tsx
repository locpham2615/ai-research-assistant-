"use client";

import React, { useEffect, useRef } from "react";
import {
  Activity,
  BookOpenCheck,
  CheckCircle2,
  Clock,
  Cpu,
  FileSearch,
  GitBranch,
  Layers,
  Tag,
  X,
  XCircle,
  Zap,
} from "lucide-react";
import type {
  AgentExecutionTrace,
  AgentStepTrace,
  Citation,
  MessageResponse,
} from "@/types/api";

export type InspectorTab = "citations" | "trace";

interface InspectorPanelProps {
  isOpen: boolean;
  activeTab: InspectorTab;
  onChangeTab: (tab: InspectorTab) => void;
  onClose: () => void;
  inspectedMessage: MessageResponse | null;
  highlightedCitationIndex: number | null;
  onSelectCitationIndex: (index: number) => void;
  liveSteps: AgentStepTrace[];
  isStreaming: boolean;
}

function formatNodeBadge(node: string): { label: string; colorClass: string } {
  switch (node) {
    case "plan":
      return {
        label: "Plan",
        colorClass: "bg-indigo-50 text-indigo-700 border-indigo-200",
      };
    case "retrieve":
      return {
        label: "Retrieve",
        colorClass: "bg-sky-50 text-sky-700 border-sky-200",
      };
    case "grade":
      return {
        label: "Grade",
        colorClass: "bg-amber-50 text-amber-800 border-amber-200",
      };
    case "synthesize":
      return {
        label: "Synthesize",
        colorClass: "bg-emerald-50 text-emerald-700 border-emerald-200",
      };
    case "arxiv_discovery":
      return {
        label: "arXiv Discovery",
        colorClass: "bg-purple-50 text-purple-700 border-purple-200",
      };
    case "abstain":
      return {
        label: "Abstain",
        colorClass: "bg-rose-50 text-rose-700 border-rose-200",
      };
    case "single_shot_rag":
      return {
        label: "Single-Shot RAG",
        colorClass: "bg-blue-50 text-blue-700 border-blue-200",
      };
    default:
      return {
        label: node,
        colorClass: "bg-slate-100 text-slate-700 border-slate-200",
      };
  }
}

export function InspectorPanel({
  isOpen,
  activeTab,
  onChangeTab,
  onClose,
  inspectedMessage,
  highlightedCitationIndex,
  onSelectCitationIndex,
  liveSteps,
  isStreaming,
}: InspectorPanelProps) {
  const citationRefs = useRef<Map<number, HTMLDivElement>>(new Map());

  useEffect(() => {
    if (!isOpen || activeTab !== "citations" || highlightedCitationIndex === null) {
      return;
    }
    const targetEl = citationRefs.current.get(highlightedCitationIndex);
    if (targetEl) {
      targetEl.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [isOpen, activeTab, highlightedCitationIndex, inspectedMessage?.message_id]);

  if (!isOpen) {
    return null;
  }

  const citations: Citation[] = inspectedMessage?.citations ?? [];
  const trace: AgentExecutionTrace | null =
    inspectedMessage?.agent_trace ?? null;

  return (
    <aside className="flex h-full w-96 shrink-0 flex-col border-l border-slate-200 bg-white">
      {/* Header + Tabs */}
      <div className="border-b border-slate-200 px-4 pt-3.5 pb-0">
        <div className="mb-3 flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-slate-900">
              Bảng Kiểm chứng Học thuật
            </h2>
            <p className="text-[11px] text-slate-500">
              Trích dẫn nguyên văn & lộ trình suy luận Agent
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            title="Đóng bảng kiểm chứng"
            className="rounded-lg p-1.5 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="grid grid-cols-2 gap-1">
          <button
            type="button"
            onClick={() => onChangeTab("citations")}
            className={`flex items-center justify-center gap-1.5 border-b-2 py-2 text-xs font-semibold transition cursor-pointer ${
              activeTab === "citations"
                ? "border-blue-700 text-blue-700"
                : "border-transparent text-slate-500 hover:text-slate-800"
            }`}
          >
            <BookOpenCheck className="h-3.5 w-3.5" />
            <span>Citations ({citations.length})</span>
          </button>

          <button
            type="button"
            onClick={() => onChangeTab("trace")}
            className={`flex items-center justify-center gap-1.5 border-b-2 py-2 text-xs font-semibold transition cursor-pointer ${
              activeTab === "trace"
                ? "border-blue-700 text-blue-700"
                : "border-transparent text-slate-500 hover:text-slate-800"
            }`}
          >
            <Activity className="h-3.5 w-3.5" />
            <span>Agent Trace</span>
          </button>
        </div>
      </div>

      {/* Panel Body */}
      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {activeTab === "citations" ? (
          citations.length === 0 ? (
            <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-slate-200 bg-slate-50/70 px-4 py-10 text-center">
              <FileSearch className="mb-2.5 h-8 w-8 text-slate-400" />
              <p className="text-xs font-medium text-slate-700">
                Chưa có nguồn trích dẫn hiển thị
              </p>
              <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
                {inspectedMessage
                  ? inspectedMessage.is_grounded === false
                    ? "Câu trả lời đang chọn đã từ chối kết luận (Abstained) do không đủ bằng chứng trong kho tài liệu."
                    : "Câu trả lời này không đính kèm nguồn trích dẫn."
                  : "Hãy chọn một câu trả lời hoặc bấm vào nhãn [1], [2] trong đoạn hội thoại để đối chiếu đoạn trích gốc."}
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {citations.map((cit) => {
                const isHighlighted = highlightedCitationIndex === cit.index;
                return (
                  <div
                    key={`inspector-cit-${cit.index}-${cit.doc_id}-${cit.chunk_index}`}
                    ref={(el) => {
                      if (el) {
                        citationRefs.current.set(cit.index, el);
                      } else {
                        citationRefs.current.delete(cit.index);
                      }
                    }}
                    onClick={() => onSelectCitationIndex(cit.index)}
                    className={`rounded-xl border p-3.5 transition cursor-pointer ${
                      isHighlighted
                        ? "border-blue-500 bg-blue-50/50 shadow-sm ring-2 ring-blue-500/20"
                        : "border-slate-200 bg-white hover:border-slate-300"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <span
                        className={`inline-flex shrink-0 items-center rounded-md px-2 py-0.5 font-mono text-xs font-bold ${
                          isHighlighted
                            ? "bg-blue-700 text-white"
                            : "bg-blue-50 text-blue-700 border border-blue-200"
                        }`}
                      >
                        [{cit.index}]
                      </span>
                      <span className="truncate font-mono text-[10px] text-slate-500">
                        {cit.doc_id} • chunk #{cit.chunk_index}
                      </span>
                    </div>

                    <h3 className="mt-2 text-xs font-semibold leading-snug text-slate-900">
                      {cit.title || cit.doc_id}
                    </h3>

                    {cit.section && (
                      <p className="mt-1 inline-block rounded bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-600">
                        Mục: {cit.section}
                      </p>
                    )}

                    <blockquote className="mt-2.5 rounded-lg border-l-2 border-blue-400 bg-slate-50 p-2.5 text-xs leading-relaxed text-slate-700">
                      {cit.snippet}
                    </blockquote>
                  </div>
                );
              })}
            </div>
          )
        ) : /* Tab: Agent Trace */ isStreaming && liveSteps.length > 0 ? (
          <div className="space-y-4">
            <div className="rounded-xl border border-blue-200 bg-blue-50/60 p-3 text-xs text-blue-900">
              <p className="font-semibold">Đang thực thi luồng suy luận...</p>
              <p className="mt-0.5 text-[11px] text-blue-700">
                Các bước LangGraph đang được cập nhật trực tiếp qua SSE.
              </p>
            </div>
            <div className="space-y-2">
              {liveSteps.map((step, idx) => {
                const badge = formatNodeBadge(step.node);
                return (
                  <div
                    key={`live-trace-${idx}-${step.node}`}
                    className="rounded-xl border border-slate-200 bg-slate-50/70 p-3 text-xs"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-1.5">
                        <span
                          className={`rounded-md border px-2 py-0.5 text-[11px] font-semibold ${badge.colorClass}`}
                        >
                          {badge.label}
                        </span>
                        <span className="font-mono text-[10px] text-slate-500">
                          Hop {step.hop}
                        </span>
                      </div>
                      <span className="font-mono text-[11px] font-medium text-slate-600">
                        {Math.round(step.latency_ms)} ms
                      </span>
                    </div>
                    {step.detail && (
                      <p className="mt-1.5 font-mono text-[11px] text-slate-600">
                        {step.detail}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ) : !trace ? (
          <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-slate-200 bg-slate-50/70 px-4 py-10 text-center">
            <Cpu className="mb-2.5 h-8 w-8 text-slate-400" />
            <p className="text-xs font-medium text-slate-700">
              Không có dữ liệu Agent Trace
            </p>
            <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
              {inspectedMessage
                ? "Câu trả lời này được chạy ở chế độ Single-Shot RAG nên không sinh đồ thị bước Agentic."
                : "Hãy gửi câu hỏi ở chế độ Agentic RAG để theo dõi kế hoạch phân rã truy vấn, chấm điểm bằng chứng và độ trễ từng bước."}
            </p>
          </div>
        ) : (
          <div className="space-y-5">
            {/* Summary Metrics Grid */}
            <section>
              <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                Tổng quan Thực thi
              </h3>
              <div className="grid grid-cols-2 gap-2">
                <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-2.5">
                  <span className="flex items-center gap-1 text-[10px] font-medium text-slate-500">
                    <GitBranch className="h-3 w-3 text-blue-600" />
                    Chiến lược (Route)
                  </span>
                  <p className="mt-1 font-mono text-xs font-semibold text-slate-900">
                    {trace.route}
                  </p>
                </div>

                <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-2.5">
                  <span className="flex items-center gap-1 text-[10px] font-medium text-slate-500">
                    <Layers className="h-3 w-3 text-sky-600" />
                    Số vòng truy xuất (Hops)
                  </span>
                  <p className="mt-1 font-mono text-xs font-semibold text-slate-900">
                    {trace.hops_executed}
                  </p>
                </div>

                <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-2.5">
                  <span className="flex items-center gap-1 text-[10px] font-medium text-slate-500">
                    <Cpu className="h-3 w-3 text-indigo-600" />
                    Số lệnh gọi LLM
                  </span>
                  <p className="mt-1 font-mono text-xs font-semibold text-slate-900">
                    {trace.llm_calls}
                  </p>
                </div>

                <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-2.5">
                  <span className="flex items-center gap-1 text-[10px] font-medium text-slate-500">
                    <Zap className="h-3 w-3 text-emerald-600" />
                    Fast-Path Grader
                  </span>
                  <p className="mt-1 font-mono text-xs font-semibold text-slate-900">
                    {trace.used_fast_path ? "Bật (Tiết kiệm LLM)" : "Không"}
                  </p>
                </div>
              </div>
            </section>

            {/* Query Plan & Sub-queries */}
            {trace.plan && (
              <section>
                <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                  Kế hoạch Truy vấn ({trace.plan.sub_queries.length} sub-queries)
                </h3>
                {trace.plan.reasoning && (
                  <p className="mb-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs leading-relaxed text-slate-600">
                    {trace.plan.reasoning}
                  </p>
                )}
                <div className="space-y-2">
                  {trace.plan.sub_queries.map((sq) => (
                    <div
                      key={sq.sub_query_id}
                      className="rounded-xl border border-slate-200 bg-white p-3"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="rounded bg-indigo-50 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-indigo-700 border border-indigo-200">
                          {sq.sub_query_id}
                        </span>
                        {sq.target_aspect && (
                          <span className="truncate text-[11px] font-medium text-slate-500">
                            {sq.target_aspect}
                          </span>
                        )}
                      </div>
                      <p className="mt-1.5 text-xs font-medium text-slate-800">
                        {sq.search_query}
                      </p>
                      {sq.bm25_keywords.length > 0 && (
                        <div className="mt-2 flex flex-wrap items-center gap-1">
                          <Tag className="h-3 w-3 text-slate-400 mr-0.5" />
                          {sq.bm25_keywords.map((kw, i) => (
                            <span
                              key={`${sq.sub_query_id}-kw-${i}`}
                              className="rounded-md bg-slate-100 px-1.5 py-0.5 font-mono text-[10px] text-slate-600"
                            >
                              {kw}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* Evidence Grades */}
            {trace.grades && trace.grades.length > 0 && (
              <section>
                <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                  Đánh giá Bằng chứng (Grades)
                </h3>
                <div className="space-y-2">
                  {trace.grades.map((grade, idx) => (
                    <div
                      key={`grade-${idx}`}
                      className="rounded-xl border border-slate-200 bg-white p-3 text-xs"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-[11px] font-semibold text-slate-600">
                          Lần chấm #{idx + 1}
                        </span>
                        {grade.is_sufficient ? (
                          <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 text-[10px] font-semibold text-emerald-700 border border-emerald-200">
                            <CheckCircle2 className="h-3 w-3" />
                            Đủ bằng chứng
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-2 py-0.5 text-[10px] font-semibold text-amber-800 border border-amber-200">
                            <XCircle className="h-3 w-3" />
                            Thiếu khía cạnh
                          </span>
                        )}
                      </div>

                      {grade.rationale && (
                        <p className="mt-1.5 leading-relaxed text-slate-700">
                          {grade.rationale}
                        </p>
                      )}

                      {grade.missing_aspects.length > 0 && (
                        <div className="mt-2">
                          <span className="text-[11px] font-medium text-amber-800">
                            Khía cạnh còn thiếu:
                          </span>
                          <ul className="mt-1 list-disc space-y-0.5 pl-4 text-[11px] text-slate-600">
                            {grade.missing_aspects.map((aspect, aIdx) => (
                              <li key={`missing-${idx}-${aIdx}`}>{aspect}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {grade.follow_up_queries.length > 0 && (
                        <div className="mt-2 rounded-lg bg-slate-50 p-2 border border-slate-100">
                          <span className="text-[10px] font-semibold uppercase text-slate-500">
                            Truy vấn bổ sung:
                          </span>
                          <ul className="mt-1 space-y-1">
                            {grade.follow_up_queries.map((fq, fIdx) => (
                              <li
                                key={`fq-${idx}-${fIdx}`}
                                className="font-mono text-[11px] text-slate-700"
                              >
                                • {fq.search_query}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* Step Latency Timeline */}
            {trace.steps && trace.steps.length > 0 && (
              <section>
                <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                  Timeline các bước ({trace.steps.length} bước)
                </h3>
                <ol className="relative ml-2 space-y-2.5 border-l border-slate-200 pl-4">
                  {trace.steps.map((step, sIdx) => {
                    const badge = formatNodeBadge(step.node);
                    return (
                      <li key={`step-${sIdx}-${step.node}`} className="relative">
                        <span className="absolute -left-[21px] top-1.5 h-2.5 w-2.5 rounded-full border-2 border-white bg-blue-600" />
                        <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-2.5">
                          <div className="flex items-center justify-between gap-2">
                            <div className="flex items-center gap-1.5">
                              <span
                                className={`rounded border px-1.5 py-0.5 text-[10px] font-semibold ${badge.colorClass}`}
                              >
                                {badge.label}
                              </span>
                              <span className="font-mono text-[10px] text-slate-500">
                                Hop {step.hop}
                              </span>
                            </div>
                            <span className="inline-flex items-center gap-1 font-mono text-[11px] font-medium text-slate-600">
                              <Clock className="h-3 w-3 text-slate-400" />
                              {Math.round(step.latency_ms)} ms
                            </span>
                          </div>
                          {step.detail && (
                            <p className="mt-1.5 font-mono text-[11px] text-slate-600 break-words">
                              {step.detail}
                            </p>
                          )}
                        </div>
                      </li>
                    );
                  })}
                </ol>
              </section>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}
