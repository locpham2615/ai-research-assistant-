"use client";

import React, { useState } from "react";
import {
  Activity,
  BookOpen,
  Check,
  FileText,
  FolderKanban,
  MessageSquare,
  MessageSquarePlus,
  Pencil,
  Plus,
  RefreshCw,
  Trash2,
  X,
} from "lucide-react";
import type {
  DocumentItem,
  HealthResponse,
  SessionResponse,
  TopicItem,
} from "@/types/api";

interface SidebarProps {
  health: HealthResponse | null;
  healthLoading: boolean;
  onRefreshHealth: () => void;
  topics: TopicItem[];
  selectedTopicId: string;
  onSelectTopic: (topicId: string) => void;
  onOpenAddPaper: () => void;
  sessions: SessionResponse[];
  activeSessionId: string | null;
  onSelectSession: (session: SessionResponse) => void;
  onNewSession: () => void;
  onRenameSession: (sessionId: string, newTitle: string) => Promise<void>;
  onDeleteSession: (sessionId: string) => Promise<void>;
  documents: DocumentItem[];
  documentsLoading: boolean;
  isStreaming: boolean;
}

function renderStatusDot(ok: boolean | undefined) {
  return (
    <span
      className={`inline-block h-2 w-2 rounded-full ${
        ok ? "bg-emerald-500" : "bg-rose-500"
      }`}
    />
  );
}

function getDocumentStatusStyle(status: string): {
  label: string;
  className: string;
} {
  const normalized = status.toLowerCase();
  if (
    normalized === "completed" ||
    normalized === "indexed" ||
    normalized === "ready" ||
    normalized === "ok" ||
    normalized === "success"
  ) {
    return {
      label: status,
      className: "bg-emerald-50 text-emerald-700 border-emerald-200",
    };
  }
  if (normalized === "skipped") {
    return {
      label: status,
      className: "bg-slate-100 text-slate-600 border-slate-200",
    };
  }
  if (normalized === "processing" || normalized === "pending") {
    return {
      label: status,
      className: "bg-blue-50 text-blue-700 border-blue-200",
    };
  }
  return {
    label: status || "unknown",
    className: "bg-rose-50 text-rose-700 border-rose-200",
  };
}

export function Sidebar({
  health,
  healthLoading,
  onRefreshHealth,
  topics,
  selectedTopicId,
  onSelectTopic,
  onOpenAddPaper,
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onRenameSession,
  onDeleteSession,
  documents,
  documentsLoading,
  isStreaming,
}: SidebarProps) {
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState<string>("");
  const [savingSessionId, setSavingSessionId] = useState<string | null>(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(
    null
  );

  const isOnline = health?.status === "ok";

  const handleStartEdit = (
    e: React.MouseEvent,
    session: SessionResponse
  ): void => {
    e.stopPropagation();
    setConfirmDeleteId(null);
    setEditingSessionId(session.session_id);
    setEditingTitle(session.title);
  };

  const handleCancelEdit = (e?: React.MouseEvent): void => {
    e?.stopPropagation();
    setEditingSessionId(null);
    setEditingTitle("");
  };

  const handleConfirmRename = async (
    e: React.MouseEvent | React.FormEvent,
    sessionId: string
  ): Promise<void> => {
    e.preventDefault();
    e.stopPropagation();
    const trimmed = editingTitle.trim();
    if (!trimmed) {
      return;
    }
    setSavingSessionId(sessionId);
    try {
      await onRenameSession(sessionId, trimmed);
      setEditingSessionId(null);
      setEditingTitle("");
    } finally {
      setSavingSessionId(null);
    }
  };

  const handleConfirmDelete = async (
    e: React.MouseEvent,
    sessionId: string
  ): Promise<void> => {
    e.stopPropagation();
    setDeletingSessionId(sessionId);
    try {
      await onDeleteSession(sessionId);
      setConfirmDeleteId(null);
    } finally {
      setDeletingSessionId(null);
    }
  };

  const hasSelectedTopicInList = topics.some((t) => t.id === selectedTopicId);

  return (
    <aside className="flex h-full w-80 shrink-0 flex-col border-r border-slate-200 bg-slate-100/75 select-none">
      {/* App Title + Health Status */}
      <div className="border-b border-slate-200 bg-white px-4 py-3.5">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-blue-700 text-white shadow-xs">
              <BookOpen className="h-4 w-4" />
            </div>
            <div className="min-w-0">
              <h1 className="truncate text-sm font-semibold tracking-tight text-slate-900">
                AI Research Assistant
              </h1>
              <p className="truncate text-[11px] text-slate-500">
                Citation-Grounded Agentic RAG
              </p>
            </div>
          </div>

          {/* Health Badge + Rich Tooltip */}
          <div className="group relative shrink-0">
            <button
              type="button"
              onClick={onRefreshHealth}
              title="Bấm để kiểm tra lại trạng thái dịch vụ"
              className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium transition ${
                isOnline
                  ? "border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-100"
                  : "border-amber-200 bg-amber-50 text-amber-800 hover:bg-amber-100"
              }`}
            >
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  isOnline ? "bg-emerald-500" : "bg-amber-500"
                }`}
              />
              <span>{isOnline ? "Online" : "Degraded"}</span>
              <RefreshCw
                className={`h-2.5 w-2.5 opacity-70 ${
                  healthLoading ? "animate-spin" : ""
                }`}
              />
            </button>

            {/* Detailed Service Tooltip */}
            <div className="pointer-events-none invisible absolute right-0 top-full z-40 mt-1.5 w-60 rounded-xl border border-slate-200 bg-white p-3 text-xs shadow-lg opacity-0 transition-all duration-150 group-hover:visible group-hover:opacity-100 group-focus-within:visible group-focus-within:opacity-100">
              <div className="mb-2 flex items-center justify-between border-b border-slate-100 pb-1.5">
                <span className="flex items-center gap-1.5 font-semibold text-slate-800">
                  <Activity className="h-3.5 w-3.5 text-blue-600" />
                  Trạng thái hạ tầng
                </span>
                <span className="font-mono text-[10px] text-slate-400">
                  /api/health
                </span>
              </div>

              {health ? (
                <div className="space-y-1.5 text-[11px]">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-600">SQLite + FTS5</span>
                    <span className="flex items-center gap-1.5 font-medium text-slate-800">
                      {renderStatusDot(health.services.sqlite_ok)}
                      {health.services.sqlite_ok ? "Sẵn sàng" : "Lỗi"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-600">Qdrant Vector DB</span>
                    <span className="flex items-center gap-1.5 font-medium text-slate-800">
                      {renderStatusDot(health.services.qdrant_ok)}
                      {health.services.qdrant_ok ? "Sẵn sàng" : "Mất kết nối"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-600">GROBID Parser</span>
                    <span className="flex items-center gap-1.5 font-medium text-slate-800">
                      {renderStatusDot(health.services.grobid_ok)}
                      {health.services.grobid_ok ? "Sẵn sàng" : "Tắt / Chờ"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-600">Local LLM</span>
                    <span className="flex items-center gap-1.5 font-medium text-slate-800">
                      {renderStatusDot(health.services.local_llm_ok)}
                      {health.services.local_llm_ok ? "Sẵn sàng" : "Mất kết nối"}
                    </span>
                  </div>
                  {health.local_llm_model && (
                    <div className="mt-2 truncate rounded-md bg-slate-50 px-2 py-1 font-mono text-[10px] text-slate-600 border border-slate-100">
                      Model: {health.local_llm_model}
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-[11px] leading-relaxed text-rose-600">
                  Không kết nối được tới FastAPI backend. Hãy kiểm tra dịch vụ
                  tại cổng 8000.
                </p>
              )}
            </div>
          </div>
        </div>

        {/* New Research Session Button */}
        <button
          type="button"
          onClick={onNewSession}
          disabled={isStreaming}
          className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg bg-blue-700 px-3 py-2 text-xs font-medium text-white shadow-xs transition hover:bg-blue-800 disabled:cursor-not-allowed disabled:opacity-60"
        >
          <MessageSquarePlus className="h-4 w-4" />
          <span>Phiên nghiên cứu mới</span>
        </button>
      </div>

      {/* Topic Selector + Add Paper */}
      <div className="border-b border-slate-200 bg-slate-50/90 px-4 py-3">
        <div className="mb-1.5 flex items-center justify-between">
          <label
            htmlFor="topic-selector"
            className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-500"
          >
            <FolderKanban className="h-3.5 w-3.5 text-slate-400" />
            Chủ đề tài liệu (Topic)
          </label>
          <button
            type="button"
            onClick={onOpenAddPaper}
            className="inline-flex items-center gap-1 rounded-md border border-blue-200 bg-blue-50 px-2 py-0.5 text-[11px] font-medium text-blue-700 transition hover:bg-blue-100"
          >
            <Plus className="h-3 w-3" />
            <span>Add Paper</span>
          </button>
        </div>

        <select
          id="topic-selector"
          value={selectedTopicId}
          onChange={(e) => onSelectTopic(e.target.value)}
          className="w-full rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-800 shadow-2xs focus:border-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
        >
          {!hasSelectedTopicInList && selectedTopicId && (
            <option value={selectedTopicId}>
              {selectedTopicId} ({documents.length} bài báo)
            </option>
          )}
          {topics.map((topic) => (
            <option key={topic.id} value={topic.id}>
              {topic.name || topic.id} ({topic.document_count} bài báo)
            </option>
          ))}
        </select>
      </div>

      {/* Scrollable Split Sections: Sessions + Papers */}
      <div className="flex min-h-0 flex-1 flex-col divide-y divide-slate-200">
        {/* Sessions List */}
        <section className="flex min-h-0 flex-1 flex-col px-3 py-3">
          <div className="mb-2 flex items-center justify-between px-1">
            <h2 className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              <MessageSquare className="h-3.5 w-3.5 text-slate-400" />
              Phiên hội thoại ({sessions.length})
            </h2>
          </div>

          <div className="min-h-0 flex-1 space-y-1 overflow-y-auto pr-0.5">
            {sessions.length === 0 ? (
              <div className="rounded-lg border border-dashed border-slate-200 bg-white/60 px-3 py-4 text-center text-xs text-slate-500">
                Chưa có phiên hội thoại nào. Hãy đặt câu hỏi đầu tiên để bắt đầu.
              </div>
            ) : (
              sessions.map((session) => {
                const isActive = session.session_id === activeSessionId;
                const isEditing = editingSessionId === session.session_id;
                const isConfirmingDelete =
                  confirmDeleteId === session.session_id;

                return (
                  <div
                    key={session.session_id}
                    onClick={() => {
                      if (!isEditing && !isStreaming) {
                        onSelectSession(session);
                      }
                    }}
                    className={`group rounded-lg border px-2.5 py-2 text-xs transition ${
                      isActive
                        ? "border-blue-200 bg-white text-slate-900 shadow-2xs"
                        : "border-transparent bg-transparent text-slate-700 hover:border-slate-200 hover:bg-white/70"
                    } ${isStreaming ? "cursor-not-allowed opacity-75" : "cursor-pointer"}`}
                  >
                    {isEditing ? (
                      <form
                        onSubmit={(e) =>
                          void handleConfirmRename(e, session.session_id)
                        }
                        onClick={(e) => e.stopPropagation()}
                        className="flex items-center gap-1"
                      >
                        <input
                          type="text"
                          value={editingTitle}
                          onChange={(e) => setEditingTitle(e.target.value)}
                          maxLength={120}
                          autoFocus
                          onKeyDown={(e) => {
                            if (e.key === "Escape") {
                              handleCancelEdit();
                            }
                          }}
                          className="min-w-0 flex-1 rounded border border-blue-400 bg-white px-2 py-1 text-xs text-slate-900 focus:outline-none"
                        />
                        <button
                          type="submit"
                          disabled={savingSessionId === session.session_id}
                          title="Lưu tên mới"
                          className="rounded p-1 text-emerald-700 hover:bg-emerald-50"
                        >
                          <Check className="h-3.5 w-3.5" />
                        </button>
                        <button
                          type="button"
                          onClick={handleCancelEdit}
                          title="Hủy"
                          className="rounded p-1 text-slate-500 hover:bg-slate-100"
                        >
                          <X className="h-3.5 w-3.5" />
                        </button>
                      </form>
                    ) : isConfirmingDelete ? (
                      <div
                        onClick={(e) => e.stopPropagation()}
                        className="flex items-center justify-between gap-2 py-0.5"
                      >
                        <span className="truncate text-[11px] font-medium text-rose-700">
                          Xóa phiên này?
                        </span>
                        <div className="flex items-center gap-1 shrink-0">
                          <button
                            type="button"
                            disabled={deletingSessionId === session.session_id}
                            onClick={(e) =>
                              void handleConfirmDelete(e, session.session_id)
                            }
                            className="rounded bg-rose-600 px-2 py-0.5 text-[11px] font-medium text-white hover:bg-rose-700 disabled:opacity-50"
                          >
                            Xóa
                          </button>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setConfirmDeleteId(null);
                            }}
                            className="rounded border border-slate-200 bg-white px-2 py-0.5 text-[11px] text-slate-600 hover:bg-slate-50"
                          >
                            Hủy
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-start justify-between gap-1.5">
                        <div className="min-w-0 flex-1">
                          <p
                            className={`truncate leading-snug ${
                              isActive ? "font-semibold" : "font-medium"
                            }`}
                            title={session.title}
                          >
                            {session.title}
                          </p>
                          <div className="mt-1 flex items-center gap-2 text-[10px] text-slate-400">
                            <span>{session.message_count} tin nhắn</span>
                            {session.topic_id && (
                              <span className="truncate rounded bg-slate-100 px-1.5 py-0.2 font-mono text-slate-500">
                                {session.topic_id}
                              </span>
                            )}
                          </div>
                        </div>

                        <div className="flex shrink-0 items-center gap-0.5 opacity-0 transition group-hover:opacity-100 focus-within:opacity-100">
                          <button
                            type="button"
                            title="Đổi tên phiên"
                            onClick={(e) => handleStartEdit(e, session)}
                            className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
                          >
                            <Pencil className="h-3 w-3" />
                          </button>
                          <button
                            type="button"
                            title="Xóa phiên"
                            onClick={(e) => {
                              e.stopPropagation();
                              setEditingSessionId(null);
                              setConfirmDeleteId(session.session_id);
                            }}
                            className="rounded p-1 text-slate-400 hover:bg-rose-50 hover:text-rose-600"
                          >
                            <Trash2 className="h-3 w-3" />
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </section>

        {/* Papers List in Current Topic */}
        <section className="flex min-h-0 flex-1 flex-col px-3 py-3">
          <div className="mb-2 flex items-center justify-between px-1">
            <h2 className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              <FileText className="h-3.5 w-3.5 text-slate-400" />
              Bài báo đã nạp ({documents.length})
            </h2>
            {documentsLoading && (
              <RefreshCw className="h-3 w-3 animate-spin text-slate-400" />
            )}
          </div>

          <div className="min-h-0 flex-1 space-y-1.5 overflow-y-auto pr-0.5">
            {documents.length === 0 ? (
              <div className="rounded-lg border border-dashed border-slate-200 bg-white/60 px-3 py-4 text-center text-xs text-slate-500">
                Chưa có bài báo nào trong chủ đề{" "}
                <span className="font-mono font-medium text-slate-700">
                  {selectedTopicId || "này"}
                </span>
                . Bấm <strong>+ Add Paper</strong> để nạp từ arXiv hoặc PDF.
              </div>
            ) : (
              documents.map((doc) => {
                const badge = getDocumentStatusStyle(doc.status);
                return (
                  <div
                    key={doc.id}
                    title={
                      doc.error_message
                        ? `${doc.title}\nLỗi: ${doc.error_message}`
                        : doc.title
                    }
                    className="rounded-lg border border-slate-200/80 bg-white px-2.5 py-2 shadow-2xs transition hover:border-slate-300"
                  >
                    <p className="line-clamp-2 text-xs font-medium leading-snug text-slate-800">
                      {doc.title || doc.id}
                    </p>
                    <div className="mt-1.5 flex items-center justify-between gap-2">
                      <span className="truncate font-mono text-[10px] text-slate-500">
                        {doc.id}
                      </span>
                      <span
                        className={`shrink-0 rounded-full border px-1.5 py-0.2 text-[10px] font-medium ${badge.className}`}
                      >
                        {badge.label}
                      </span>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </section>
      </div>
    </aside>
  );
}
