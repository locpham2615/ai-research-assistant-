"use client";

import React from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  AlertTriangle,
  BookMarked,
  CheckCircle2,
  GitBranch,
  Globe,
  Sparkles,
  User,
  Zap,
} from "lucide-react";
import type { Citation, MessageResponse } from "@/types/api";

interface MessageBubbleProps {
  message: MessageResponse;
  isInspected: boolean;
  highlightedCitationIndex: number | null;
  onSelectCitation: (messageId: string, citationIndex: number) => void;
  onInspectTrace: (messageId: string) => void;
}

const CITATION_GROUP_REGEX = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

function renderTextWithCitations(
  text: string,
  validIndices: Set<number>,
  activeCitationIndex: number | null,
  onClickIndex: (idx: number) => void,
  keyPrefix: string
): React.ReactNode {
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  CITATION_GROUP_REGEX.lastIndex = 0;
  while ((match = CITATION_GROUP_REGEX.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    const rawNumbers = match[1]
      .split(",")
      .map((s) => Number.parseInt(s.trim(), 10))
      .filter((n) => Number.isFinite(n) && n > 0);

    if (rawNumbers.length === 0) {
      parts.push(match[0]);
    } else {
      parts.push(
        <span
          key={`${keyPrefix}-grp-${match.index}`}
          className="mx-0.5 inline-flex flex-wrap items-center gap-0.5 align-baseline"
        >
          {rawNumbers.map((num, subIdx) => {
            const existsInList =
              validIndices.size === 0 || validIndices.has(num);
            const isHighlighted = activeCitationIndex === num;
            return (
              <button
                key={`${keyPrefix}-cit-${match?.index}-${num}-${subIdx}`}
                type="button"
                onClick={() => onClickIndex(num)}
                title={
                  existsInList
                    ? `Xem nguồn trích dẫn [${num}]`
                    : `Nguồn [${num}]`
                }
                className={`inline-flex items-center rounded px-1.5 py-0.2 font-mono text-[11px] font-semibold leading-tight transition cursor-pointer ${
                  isHighlighted
                    ? "bg-blue-700 text-white ring-2 ring-blue-300"
                    : existsInList
                      ? "border border-blue-200 bg-blue-50 text-blue-700 hover:border-blue-400 hover:bg-blue-600 hover:text-white"
                      : "border border-slate-200 bg-slate-100 text-slate-600 hover:bg-slate-200"
                }`}
              >
                [{num}]
              </button>
            );
          })}
        </span>
      );
    }

    lastIndex = match.index + match[0].length;
  }

  if (parts.length === 0) {
    return text;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts;
}

function injectCitationsIntoChildren(
  children: React.ReactNode,
  validIndices: Set<number>,
  activeCitationIndex: number | null,
  onClickIndex: (idx: number) => void,
  keyPrefix = "node"
): React.ReactNode {
  if (typeof children === "string") {
    return renderTextWithCitations(
      children,
      validIndices,
      activeCitationIndex,
      onClickIndex,
      keyPrefix
    );
  }

  if (Array.isArray(children)) {
    return children.map((child, idx) =>
      injectCitationsIntoChildren(
        child,
        validIndices,
        activeCitationIndex,
        onClickIndex,
        `${keyPrefix}-${idx}`
      )
    );
  }

  if (React.isValidElement<{ children?: React.ReactNode }>(children)) {
    const elementType = children.type;
    if (
      elementType === "code" ||
      elementType === "pre" ||
      elementType === "a"
    ) {
      return children;
    }

    if (children.props && children.props.children !== undefined) {
      return React.cloneElement(
        children,
        undefined,
        injectCitationsIntoChildren(
          children.props.children,
          validIndices,
          activeCitationIndex,
          onClickIndex,
          `${keyPrefix}-el`
        )
      );
    }
  }

  return children;
}

export function MessageBubble({
  message,
  isInspected,
  highlightedCitationIndex,
  onSelectCitation,
  onInspectTrace,
}: MessageBubbleProps) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-2xl rounded-2xl rounded-tr-sm border border-blue-800/10 bg-blue-700 px-4 py-3 text-sm leading-relaxed text-white shadow-xs">
          <div className="mb-1 flex items-center justify-end gap-1.5 text-[11px] font-medium text-blue-100">
            <span>Nhà nghiên cứu</span>
            <User className="h-3 w-3" />
          </div>
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    );
  }

  const citations: Citation[] = message.citations ?? [];
  const validIndices = new Set(citations.map((c) => c.index));
  const activeIdx = isInspected ? highlightedCitationIndex : null;
  const isGrounded = message.is_grounded !== false && citations.length > 0;
  const isExplicitlyAbstained = message.is_grounded === false;
  const trace = message.agent_trace;

  const handleCitationClick = (idx: number): void => {
    onSelectCitation(message.message_id, idx);
  };

  const markdownComponents: Components = {
    p: ({ children }) => (
      <p>
        {injectCitationsIntoChildren(
          children,
          validIndices,
          activeIdx,
          handleCitationClick,
          "p"
        )}
      </p>
    ),
    li: ({ children }) => (
      <li>
        {injectCitationsIntoChildren(
          children,
          validIndices,
          activeIdx,
          handleCitationClick,
          "li"
        )}
      </li>
    ),
    td: ({ children }) => (
      <td>
        {injectCitationsIntoChildren(
          children,
          validIndices,
          activeIdx,
          handleCitationClick,
          "td"
        )}
      </td>
    ),
    th: ({ children }) => (
      <th>
        {injectCitationsIntoChildren(
          children,
          validIndices,
          activeIdx,
          handleCitationClick,
          "th"
        )}
      </th>
    ),
    blockquote: ({ children }) => (
      <blockquote>
        {injectCitationsIntoChildren(
          children,
          validIndices,
          activeIdx,
          handleCitationClick,
          "bq"
        )}
      </blockquote>
    ),
  };

  return (
    <div className="flex justify-start">
      <article
        className={`w-full max-w-3xl rounded-2xl rounded-tl-sm border bg-white p-4 shadow-xs transition ${
          isInspected
            ? "border-blue-300 ring-2 ring-blue-500/10"
            : "border-slate-200"
        }`}
      >
        {/* Header: Grounded / Abstained Badge + Quick Trace Summary */}
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-2.5">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-800">
              <Sparkles className="h-3.5 w-3.5 text-blue-600" />
              Trợ lý Nghiên cứu
            </span>

            {isExplicitlyAbstained ? (
              <span className="inline-flex items-center gap-1 rounded-full border border-amber-200 bg-amber-50 px-2.5 py-0.5 text-[11px] font-medium text-amber-800">
                <AlertTriangle className="h-3 w-3 text-amber-600" />
                Abstained — Không đủ bằng chứng
              </span>
            ) : isGrounded ? (
              <span className="inline-flex items-center gap-1 rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-0.5 text-[11px] font-medium text-emerald-700">
                <CheckCircle2 className="h-3 w-3 text-emerald-600" />
                Grounded ({citations.length} trích dẫn)
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-slate-100 px-2.5 py-0.5 text-[11px] font-medium text-slate-600">
                Phản hồi tổng hợp
              </span>
            )}
          </div>

          {/* Short Trace Summary Pills */}
          <div className="flex flex-wrap items-center gap-1.5">
            {trace ? (
              <button
                type="button"
                onClick={() => onInspectTrace(message.message_id)}
                title="Bấm để mở chi tiết Agent Trace ở cột phải"
                className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-600 transition hover:border-blue-300 hover:bg-blue-50/70 hover:text-blue-700 cursor-pointer"
              >
                <span className="inline-flex items-center gap-1">
                  <GitBranch className="h-3 w-3 text-blue-600" />
                  route:{" "}
                  <strong className="font-mono text-slate-800">
                    {trace.route}
                  </strong>
                </span>
                <span className="text-slate-300">•</span>
                <span>
                  hops:{" "}
                  <strong className="font-mono text-slate-800">
                    {trace.hops_executed}
                  </strong>
                </span>
                {trace.used_fast_path && (
                  <>
                    <span className="text-slate-300">•</span>
                    <span className="inline-flex items-center gap-0.5 text-emerald-700">
                      <Zap className="h-3 w-3" />
                      fast-path
                    </span>
                  </>
                )}
                {trace.used_external_arxiv && (
                  <>
                    <span className="text-slate-300">•</span>
                    <span className="inline-flex items-center gap-0.5 text-indigo-700">
                      <Globe className="h-3 w-3" />
                      arXiv
                    </span>
                  </>
                )}
              </button>
            ) : (
              <span className="rounded-md border border-slate-200 bg-slate-50 px-2 py-0.5 font-mono text-[10px] text-slate-500">
                Single-Shot RAG
              </span>
            )}
          </div>
        </div>

        {/* Abstention Reason Banner when is_grounded === false */}
        {isExplicitlyAbstained && message.abstain_reason && (
          <div className="mb-3 rounded-xl border border-amber-200/90 bg-amber-50/70 px-3.5 py-2.5 text-xs text-amber-900">
            <p className="font-semibold">Lý do từ chối kết luận (Abstain Reason):</p>
            <p className="mt-0.5 font-mono text-[11px] leading-relaxed text-amber-800">
              {message.abstain_reason}
            </p>
          </div>
        )}

        {/* Markdown Content with Interactive Inline Citations */}
        <div className="academic-markdown">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={markdownComponents}
          >
            {message.content}
          </ReactMarkdown>
        </div>

        {/* Citation Chips Footer */}
        {citations.length > 0 && (
          <div className="mt-4 border-t border-slate-100 pt-3">
            <div className="mb-1.5 flex items-center gap-1.5 text-[11px] font-semibold text-slate-500">
              <BookMarked className="h-3.5 w-3.5 text-blue-600" />
              <span>Nguồn tham chiếu được xác thực ({citations.length}):</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {citations.map((cit) => {
                const isChipActive = activeIdx === cit.index;
                return (
                  <button
                    key={`${message.message_id}-chip-${cit.index}`}
                    type="button"
                    onClick={() => handleCitationClick(cit.index)}
                    title={`${cit.title} — ${cit.section || "Section"} (chunk #${cit.chunk_index})`}
                    className={`inline-flex max-w-xs items-center gap-1.5 rounded-lg border px-2.5 py-1 text-left text-xs transition cursor-pointer ${
                      isChipActive
                        ? "border-blue-600 bg-blue-700 text-white shadow-2xs"
                        : "border-slate-200 bg-slate-50 text-slate-700 hover:border-blue-300 hover:bg-blue-50 hover:text-blue-800"
                    }`}
                  >
                    <span
                      className={`shrink-0 font-mono text-[11px] font-bold ${
                        isChipActive ? "text-blue-100" : "text-blue-700"
                      }`}
                    >
                      [{cit.index}]
                    </span>
                    <span className="truncate font-medium">
                      {cit.title || cit.doc_id}
                    </span>
                    {cit.section && (
                      <span
                        className={`hidden truncate text-[10px] sm:inline ${
                          isChipActive ? "text-blue-200" : "text-slate-400"
                        }`}
                      >
                        • {cit.section}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </article>
    </div>
  );
}
