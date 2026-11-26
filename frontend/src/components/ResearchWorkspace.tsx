"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  ArrowUp,
  BookOpen,
  CheckCircle2,
  Clock,
  Compass,
  Cpu,
  Globe,
  Loader2,
  PanelRightClose,
  PanelRightOpen,
  RefreshCw,
  ShieldAlert,
  Sparkles,
  Square,
  Zap,
} from "lucide-react";
import {
  deleteSession,
  getHealth,
  getSessionMessages,
  listDocuments,
  listSessions,
  listTopics,
  renameSession,
  streamChat,
} from "@/lib/api";
import type {
  AgentStepTrace,
  ChatMode,
  ChatResponse,
  DocumentItem,
  HealthResponse,
  MessageResponse,
  SessionResponse,
  TopicItem,
} from "@/types/api";
import { AddPaperModal } from "@/components/AddPaperModal";
import { InspectorPanel, type InspectorTab } from "@/components/InspectorPanel";
import { MessageBubble } from "@/components/MessageBubble";
import { Sidebar } from "@/components/Sidebar";

const ACTIVE_SESSION_STORAGE_KEY = "ai_research_assistant_active_session_id";
const DEFAULT_TOPIC_ID = "llm_eval_v1";

interface SamplePrompt {
  badge: string;
  title: string;
  query: string;
}

const SAMPLE_PROMPTS: SamplePrompt[] = [
  {
    badge: "Multi-Paper • LoRA vs QLoRA",
    title: "So sánh vị trí gắn adapter & tối ưu bộ nhớ giữa LoRA và QLoRA",
    query:
      "Bài báo LoRA gốc và bài báo QLoRA khác nhau như thế nào trong kết luận về việc nên gắn LoRA adapters vào những ma trận trọng số nào của khối Transformer, cũng như cơ chế tối ưu bộ nhớ GPU?",
  },
  {
    badge: "Tiếng Việt • Transformer Attention",
    title: "Ý nghĩa hệ số chuẩn hóa sqrt(d_k) trong Scaled Dot-Product Attention",
    query:
      "Trong kiến trúc Transformer, tại sao công thức Scaled Dot-Product Attention lại phải chia tích vô hướng cho sqrt(d_k) trước khi đưa qua hàm softmax?",
  },
  {
    badge: "Systems • vLLM PagedAttention",
    title: "Cơ chế quản lý KV cache không liên tục của PagedAttention",
    query:
      "Cơ chế PagedAttention trong vLLM giải quyết vấn đề phân mảnh bộ nhớ KV cache và chia sẻ các physical blocks giữa nhiều request như thế nào?",
  },
  {
    badge: "Kiểm thử Abstention • Llama 3 405B",
    title: "Kiểm chứng khả năng từ chối trả lời khi tài liệu không đề cập",
    query:
      "Mô hình Llama 3 405B đạt điểm số MMLU bao nhiêu và sử dụng cấu hình lượng tử hóa FP8 trong quá trình huấn luyện như thế nào theo kho tài liệu hiện tại?",
  },
];

const PIPELINE_STAGES: Array<{ key: string; label: string }> = [
  { key: "plan", label: "Plan" },
  { key: "retrieve", label: "Retrieve" },
  { key: "grade", label: "Grade" },
  { key: "synthesize", label: "Synthesize" },
];

export function ResearchWorkspace() {
  // Backend data state
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);
  const [backendError, setBackendError] = useState<string>("");

  const [topics, setTopics] = useState<TopicItem[]>([]);
  const [selectedTopicId, setSelectedTopicId] =
    useState<string>(DEFAULT_TOPIC_ID);

  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState<boolean>(false);

  const [sessions, setSessions] = useState<SessionResponse[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  const [messages, setMessages] = useState<MessageResponse[]>([]);
  const [messagesLoading, setMessagesLoading] = useState<boolean>(false);

  // Composer & Streaming state
  const [queryInput, setQueryInput] = useState<string>("");
  const [mode, setMode] = useState<ChatMode>("agentic");
  const [allowExternalArxiv, setAllowExternalArxiv] = useState<boolean>(false);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [liveSteps, setLiveSteps] = useState<AgentStepTrace[]>([]);
  const [chatError, setChatError] = useState<string>("");

  // Inspector & Modal state
  const [isInspectorOpen, setIsInspectorOpen] = useState<boolean>(true);
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("citations");
  const [inspectedMessageId, setInspectedMessageId] = useState<string | null>(
    null
  );
  const [highlightedCitationIndex, setHighlightedCitationIndex] = useState<
    number | null
  >(null);
  const [isAddPaperOpen, setIsAddPaperOpen] = useState<boolean>(false);

  const abortControllerRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const optimisticSeqRef = useRef<number>(0);

  const persistActiveSession = useCallback((sessionId: string | null) => {
    try {
      if (sessionId) {
        window.localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, sessionId);
      } else {
        window.localStorage.removeItem(ACTIVE_SESSION_STORAGE_KEY);
      }
    } catch {
      // Ignore storage errors in restricted environments
    }
  }, []);

  const refreshHealth = useCallback(async () => {
    setHealthLoading(true);
    try {
      const h = await getHealth();
      setHealth(h);
      setBackendError("");
    } catch (err) {
      setHealth(null);
      setBackendError(
        err instanceof Error
          ? err.message
          : "Không thể kết nối tới FastAPI backend."
      );
    } finally {
      setHealthLoading(false);
    }
  }, []);

  const refreshDocumentsForTopic = useCallback(async (topicId: string) => {
    setDocumentsLoading(true);
    try {
      const docs = await listDocuments(topicId || undefined);
      setDocuments(docs);
    } catch {
      setDocuments([]);
    } finally {
      setDocumentsLoading(false);
    }
  }, []);

  const refreshTopicsAndDocuments = useCallback(
    async (preferredTopicId?: string) => {
      const fetchedTopics = await listTopics();
      setTopics(fetchedTopics);

      let targetTopic = preferredTopicId || selectedTopicId;
      if (fetchedTopics.length > 0) {
        const hasPreferred =
          preferredTopicId &&
          fetchedTopics.some((t) => t.id === preferredTopicId);
        const hasDefault = fetchedTopics.some((t) => t.id === DEFAULT_TOPIC_ID);
        const hasCurrent = fetchedTopics.some((t) => t.id === targetTopic);

        if (hasPreferred && preferredTopicId) {
          targetTopic = preferredTopicId;
        } else if (!hasCurrent) {
          targetTopic = hasDefault ? DEFAULT_TOPIC_ID : fetchedTopics[0].id;
        }
      } else if (!targetTopic) {
        targetTopic = DEFAULT_TOPIC_ID;
      }

      setSelectedTopicId(targetTopic);
      await refreshDocumentsForTopic(targetTopic);
    },
    [refreshDocumentsForTopic, selectedTopicId]
  );

  // Initial workspace bootstrap
  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      setHealthLoading(true);
      try {
        const [healthRes, topicsRes, sessionsRes] = await Promise.allSettled([
          getHealth(),
          listTopics(),
          listSessions({ limit: 100 }),
        ]);

        if (cancelled) {
          return;
        }

        if (healthRes.status === "fulfilled") {
          setHealth(healthRes.value);
          setBackendError("");
        } else {
          setHealth(null);
          setBackendError(
            healthRes.reason instanceof Error
              ? healthRes.reason.message
              : "Không thể kết nối tới FastAPI backend."
          );
        }

        let initialTopic = DEFAULT_TOPIC_ID;
        if (topicsRes.status === "fulfilled") {
          const loadedTopics = topicsRes.value;
          setTopics(loadedTopics);
          if (loadedTopics.length > 0) {
            const defaultMatch = loadedTopics.find(
              (t) => t.id === DEFAULT_TOPIC_ID
            );
            initialTopic = defaultMatch ? defaultMatch.id : loadedTopics[0].id;
          }
          setSelectedTopicId(initialTopic);
        }

        if (sessionsRes.status === "fulfilled") {
          const loadedSessions = sessionsRes.value;
          setSessions(loadedSessions);

          let savedSessionId: string | null = null;
          try {
            savedSessionId = window.localStorage.getItem(
              ACTIVE_SESSION_STORAGE_KEY
            );
          } catch {
            savedSessionId = null;
          }

          if (
            savedSessionId &&
            loadedSessions.some((s) => s.session_id === savedSessionId)
          ) {
            setActiveSessionId(savedSessionId);
            setMessagesLoading(true);
            try {
              const restoredMessages = await getSessionMessages(savedSessionId);
              if (!cancelled) {
                setMessages(restoredMessages);
              }
            } catch {
              // Ignore message restore failure
            } finally {
              if (!cancelled) {
                setMessagesLoading(false);
              }
            }
          }
        }

        try {
          setDocumentsLoading(true);
          const docs = await listDocuments(initialTopic);
          if (!cancelled) {
            setDocuments(docs);
          }
        } catch {
          if (!cancelled) {
            setDocuments([]);
          }
        } finally {
          if (!cancelled) {
            setDocumentsLoading(false);
          }
        }
      } finally {
        if (!cancelled) {
          setHealthLoading(false);
        }
      }
    }

    void bootstrap();
    return () => {
      cancelled = true;
    };
  }, []);

  // Auto-scroll chat when new messages or live steps arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, liveSteps.length, isStreaming]);

  const handleSelectTopic = async (newTopicId: string): Promise<void> => {
    setSelectedTopicId(newTopicId);
    await refreshDocumentsForTopic(newTopicId);
  };

  const handleNewSession = (): void => {
    if (isStreaming) {
      return;
    }
    setActiveSessionId(null);
    persistActiveSession(null);
    setMessages([]);
    setLiveSteps([]);
    setChatError("");
    setInspectedMessageId(null);
    setHighlightedCitationIndex(null);
  };

  const handleSelectSession = async (
    session: SessionResponse
  ): Promise<void> => {
    if (isStreaming) {
      return;
    }
    setChatError("");
    setLiveSteps([]);
    setActiveSessionId(session.session_id);
    persistActiveSession(session.session_id);
    setInspectedMessageId(null);
    setHighlightedCitationIndex(null);

    if (session.topic_id && session.topic_id !== selectedTopicId) {
      setSelectedTopicId(session.topic_id);
      void refreshDocumentsForTopic(session.topic_id);
    }

    setMessagesLoading(true);
    try {
      const loadedMessages = await getSessionMessages(session.session_id);
      setMessages(loadedMessages);
    } catch (err) {
      setChatError(
        err instanceof Error
          ? err.message
          : "Không thể tải lịch sử tin nhắn của phiên này."
      );
    } finally {
      setMessagesLoading(false);
    }
  };

  const handleRenameSession = async (
    sessionId: string,
    newTitle: string
  ): Promise<void> => {
    const updated = await renameSession(sessionId, { title: newTitle });
    setSessions((prev) =>
      prev.map((s) => (s.session_id === sessionId ? updated : s))
    );
  };

  const handleDeleteSession = async (sessionId: string): Promise<void> => {
    await deleteSession(sessionId);
    setSessions((prev) => prev.filter((s) => s.session_id !== sessionId));
    if (activeSessionId === sessionId) {
      setActiveSessionId(null);
      persistActiveSession(null);
      setMessages([]);
      setInspectedMessageId(null);
      setHighlightedCitationIndex(null);
    }
  };

  const handlePaperIngested = async (ingestedTopicId: string): Promise<void> => {
    await Promise.allSettled([
      refreshHealth(),
      refreshTopicsAndDocuments(ingestedTopicId),
    ]);
  };

  const handleStopStream = (): void => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setIsStreaming(false);
  };

  const sendQuery = async (rawQuery: string): Promise<void> => {
    const trimmed = rawQuery.trim();
    if (!trimmed || isStreaming) {
      return;
    }

    setChatError("");
    setQueryInput("");
    setLiveSteps([]);
    setHighlightedCitationIndex(null);

    optimisticSeqRef.current += 1;
    const tempUserMsgId = `temp-user-${optimisticSeqRef.current}`;
    const nowIso = new Date().toISOString();
    const optimisticUserMsg: MessageResponse = {
      message_id: tempUserMsgId,
      session_id: activeSessionId ?? "pending",
      role: "user",
      content: trimmed,
      citations: [],
      agent_trace: null,
      is_grounded: null,
      abstain_reason: "",
      created_at: nowIso,
    };

    setMessages((prev) => [...prev, optimisticUserMsg]);
    setIsStreaming(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      await streamChat(
        {
          query: trimmed,
          session_id: activeSessionId,
          topic_id: selectedTopicId || null,
          mode,
          allow_external_arxiv: mode === "agentic" ? allowExternalArxiv : false,
        },
        {
          onSession: (evt) => {
            setActiveSessionId(evt.session_id);
            persistActiveSession(evt.session_id);
            setMessages((prev) =>
              prev.map((m) =>
                m.message_id === tempUserMsgId
                  ? {
                      ...m,
                      message_id: evt.user_message_id,
                      session_id: evt.session_id,
                    }
                  : m
              )
            );
          },
          onStep: (step) => {
            setLiveSteps((prev) => [...prev, step]);
          },
          onFinal: (finalRes: ChatResponse) => {
            const assistantMsg: MessageResponse = {
              message_id: finalRes.assistant_message_id,
              session_id: finalRes.session_id,
              role: "assistant",
              content: finalRes.answer,
              citations: finalRes.citations,
              agent_trace: finalRes.agent_trace,
              is_grounded: finalRes.is_grounded,
              abstain_reason: finalRes.abstain_reason,
              created_at: new Date().toISOString(),
            };

            setActiveSessionId(finalRes.session_id);
            persistActiveSession(finalRes.session_id);
            setMessages((prev) => [...prev, assistantMsg]);
            setInspectedMessageId(assistantMsg.message_id);
            setHighlightedCitationIndex(null);

            void listSessions({ limit: 100 })
              .then((updatedSessions) => setSessions(updatedSessions))
              .catch(() => undefined);

            if (finalRes.agent_trace?.used_external_arxiv) {
              void refreshTopicsAndDocuments(selectedTopicId);
            }
          },
          onError: (errMsg) => {
            setChatError(errMsg);
          },
        },
        controller.signal
      );
    } catch (err) {
      if (!(err instanceof DOMException && err.name === "AbortError")) {
        setChatError(
          err instanceof Error
            ? err.message
            : "Đã xảy ra lỗi trong quá trình nhận phản hồi từ máy chủ."
        );
      }
    } finally {
      abortControllerRef.current = null;
      setIsStreaming(false);
    }
  };

  const handleSelectCitation = (
    messageId: string,
    citationIndex: number
  ): void => {
    setInspectedMessageId(messageId);
    setHighlightedCitationIndex(citationIndex);
    setInspectorTab("citations");
    setIsInspectorOpen(true);
  };

  const handleInspectTrace = (messageId: string): void => {
    setInspectedMessageId(messageId);
    setInspectorTab("trace");
    setIsInspectorOpen(true);
  };

  // Resolve inspected assistant message (explicitly selected, or latest assistant message)
  const inspectedMessage = useMemo<MessageResponse | null>(() => {
    const assistantMessages = messages.filter((m) => m.role === "assistant");
    if (assistantMessages.length === 0) {
      return null;
    }
    if (inspectedMessageId) {
      const found = assistantMessages.find(
        (m) => m.message_id === inspectedMessageId
      );
      if (found) {
        return found;
      }
    }
    return assistantMessages[assistantMessages.length - 1];
  }, [messages, inspectedMessageId]);

  const activeSessionTitle = useMemo(() => {
    if (!activeSessionId) {
      return "Phiên nghiên cứu mới";
    }
    const found = sessions.find((s) => s.session_id === activeSessionId);
    return found?.title || "Phiên hội thoại đang mở";
  }, [activeSessionId, sessions]);

  const completedNodeSet = useMemo(
    () => new Set(liveSteps.map((s) => s.node)),
    [liveSteps]
  );

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-50 text-slate-900">
      {/* Left Column: Sidebar */}
      <Sidebar
        health={health}
        healthLoading={healthLoading}
        onRefreshHealth={() => void refreshHealth()}
        topics={topics}
        selectedTopicId={selectedTopicId}
        onSelectTopic={(tid) => void handleSelectTopic(tid)}
        onOpenAddPaper={() => setIsAddPaperOpen(true)}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={(s) => void handleSelectSession(s)}
        onNewSession={handleNewSession}
        onRenameSession={handleRenameSession}
        onDeleteSession={handleDeleteSession}
        documents={documents}
        documentsLoading={documentsLoading}
        isStreaming={isStreaming}
      />

      {/* Center Column: Chat Workspace */}
      <main className="flex min-w-0 flex-1 flex-col bg-slate-50">
        {/* Top Bar */}
        <header className="flex h-14 shrink-0 items-center justify-between gap-3 border-b border-slate-200 bg-white px-5">
          <div className="flex min-w-0 items-center gap-2.5">
            <h2 className="truncate text-sm font-semibold text-slate-900">
              {activeSessionTitle}
            </h2>
            {selectedTopicId && (
              <span className="shrink-0 rounded-full border border-slate-200 bg-slate-100 px-2.5 py-0.5 font-mono text-[11px] font-medium text-slate-600">
                topic: {selectedTopicId}
              </span>
            )}
          </div>

          <div className="flex shrink-0 items-center gap-2">
            <button
              type="button"
              onClick={() => setIsInspectorOpen((prev) => !prev)}
              className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition cursor-pointer ${
                isInspectorOpen
                  ? "border-blue-200 bg-blue-50 text-blue-700"
                  : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
              }`}
            >
              {isInspectorOpen ? (
                <>
                  <PanelRightClose className="h-3.5 w-3.5" />
                  <span>Ẩn Inspector</span>
                </>
              ) : (
                <>
                  <PanelRightOpen className="h-3.5 w-3.5" />
                  <span>
                    Mở Inspector
                    {inspectedMessage?.citations.length
                      ? ` (${inspectedMessage.citations.length})`
                      : ""}
                  </span>
                </>
              )}
            </button>
          </div>
        </header>

        {/* Backend Offline Banner (graceful non-crashing state) */}
        {backendError && (
          <div className="flex items-center justify-between gap-3 border-b border-amber-200 bg-amber-50 px-5 py-2 text-xs text-amber-900">
            <div className="flex items-center gap-2">
              <ShieldAlert className="h-4 w-4 shrink-0 text-amber-600" />
              <span>{backendError}</span>
            </div>
            <button
              type="button"
              onClick={() => {
                void refreshHealth();
                void refreshTopicsAndDocuments(selectedTopicId);
                void listSessions({ limit: 100 })
                  .then((s) => setSessions(s))
                  .catch(() => undefined);
              }}
              className="inline-flex shrink-0 items-center gap-1 rounded-md border border-amber-300 bg-white px-2.5 py-1 text-[11px] font-semibold text-amber-800 hover:bg-amber-100"
            >
              <RefreshCw className="h-3 w-3" />
              <span>Thử kết nối lại</span>
            </button>
          </div>
        )}

        {/* Messages / Empty State Scroll Area */}
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-6">
          <div className="mx-auto max-w-4xl">
            {messagesLoading ? (
              <div className="flex flex-col items-center justify-center py-20 text-slate-500">
                <Loader2 className="mb-2 h-6 w-6 animate-spin text-blue-600" />
                <p className="text-xs font-medium">
                  Đang tải lịch sử hội thoại và trích dẫn...
                </p>
              </div>
            ) : messages.length === 0 && !isStreaming ? (
              /* Empty State with 4 Sample Prompts */
              <div className="py-6">
                <div className="mx-auto max-w-2xl text-center">
                  <div className="mx-auto mb-3 flex h-11 w-11 items-center justify-center rounded-2xl border border-blue-200 bg-blue-50 text-blue-700">
                    <BookOpen className="h-5 w-5" />
                  </div>
                  <h3 className="text-lg font-semibold tracking-tight text-slate-900">
                    Không gian Nghiên cứu Học thuật Có Kiểm chứng Trích dẫn
                  </h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-slate-600">
                    Mọi luận điểm đều được đối chiếu trực tiếp với kho bài báo khoa
                    học qua Hybrid Retrieval (Qdrant + SQLite FTS5), Cross-Encoder
                    Reranker và LangGraph Agentic Grader. Hệ thống tự động từ chối
                    trả lời (Abstain) khi không đủ bằng chứng thực nghiệm.
                  </p>
                </div>

                <div className="mt-7 grid grid-cols-1 gap-3 md:grid-cols-2">
                  {SAMPLE_PROMPTS.map((sample, index) => (
                    <button
                      key={`sample-${index}`}
                      type="button"
                      onClick={() => void sendQuery(sample.query)}
                      className="group flex flex-col justify-between rounded-2xl border border-slate-200 bg-white p-4 text-left shadow-2xs transition hover:border-blue-400 hover:shadow-sm cursor-pointer"
                    >
                      <div>
                        <span className="inline-flex items-center gap-1 rounded-md bg-slate-100 px-2 py-0.5 font-mono text-[10px] font-semibold text-slate-600 group-hover:bg-blue-50 group-hover:text-blue-700">
                          <Compass className="h-3 w-3" />
                          {sample.badge}
                        </span>
                        <h4 className="mt-2 text-xs font-semibold text-slate-900 group-hover:text-blue-800">
                          {sample.title}
                        </h4>
                        <p className="mt-1.5 line-clamp-3 text-xs leading-relaxed text-slate-600">
                          {sample.query}
                        </p>
                      </div>
                      <span className="mt-3 inline-flex items-center gap-1 text-[11px] font-semibold text-blue-700 opacity-80 group-hover:opacity-100">
                        Gửi truy vấn mẫu →
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              /* Conversation Thread */
              <div className="space-y-5">
                {messages.map((msg) => (
                  <MessageBubble
                    key={msg.message_id}
                    message={msg}
                    isInspected={inspectedMessage?.message_id === msg.message_id}
                    highlightedCitationIndex={highlightedCitationIndex}
                    onSelectCitation={handleSelectCitation}
                    onInspectTrace={handleInspectTrace}
                  />
                ))}

                {/* Live Agent Thinking Panel while Streaming */}
                {isStreaming && (
                  <div className="max-w-3xl rounded-2xl rounded-tl-sm border border-blue-200 bg-white p-4 shadow-xs">
                    <div className="flex items-center justify-between gap-2 border-b border-slate-100 pb-3">
                      <div className="flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin text-blue-700" />
                        <span className="text-xs font-semibold text-slate-900">
                          {mode === "agentic"
                            ? "Agentic RAG đang phân tích và kiểm chứng bằng chứng..."
                            : "Single-Shot RAG đang truy xuất và tổng hợp câu trả lời..."}
                        </span>
                      </div>
                      <span className="font-mono text-[10px] text-slate-400">
                        SSE Live Stream
                      </span>
                    </div>

                    {/* Node Progress Stepper for Agentic Mode */}
                    {mode === "agentic" && (
                      <div className="mt-3 flex flex-wrap items-center gap-1.5">
                        {PIPELINE_STAGES.map((stage, idx) => {
                          const isDone = completedNodeSet.has(stage.key);
                          return (
                            <React.Fragment key={stage.key}>
                              <span
                                className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[11px] font-medium ${
                                  isDone
                                    ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                                    : "border-slate-200 bg-slate-50 text-slate-500"
                                }`}
                              >
                                {isDone ? (
                                  <CheckCircle2 className="h-3 w-3 text-emerald-600" />
                                ) : (
                                  <span className="h-1.5 w-1.5 rounded-full bg-slate-300" />
                                )}
                                {stage.label}
                              </span>
                              {idx < PIPELINE_STAGES.length - 1 && (
                                <span className="text-xs text-slate-300">→</span>
                              )}
                            </React.Fragment>
                          );
                        })}
                      </div>
                    )}

                    {/* Real-time Step Rows */}
                    {liveSteps.length > 0 ? (
                      <div className="mt-3 space-y-1.5">
                        {liveSteps.map((step, idx) => (
                          <div
                            key={`stream-step-${idx}-${step.node}`}
                            className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-100 bg-slate-50 px-3 py-2 text-xs"
                          >
                            <div className="flex flex-wrap items-center gap-2 min-w-0">
                              <span className="rounded bg-blue-100/80 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-blue-800">
                                {step.node}
                              </span>
                              <span className="font-mono text-[10px] text-slate-500">
                                hop={step.hop}
                              </span>
                              {step.detail && (
                                <span className="truncate font-mono text-[11px] text-slate-700">
                                  {step.detail}
                                </span>
                              )}
                            </div>
                            <span className="inline-flex shrink-0 items-center gap-1 font-mono text-[11px] text-slate-500">
                              <Clock className="h-3 w-3" />
                              {Math.round(step.latency_ms)} ms
                            </span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="mt-2.5 text-xs text-slate-500">
                        Đang khởi tạo kế hoạch truy vấn trên mô hình ngôn ngữ...
                      </p>
                    )}
                  </div>
                )}

                {/* Stream / Chat Error Alert */}
                {chatError && (
                  <div className="flex items-start gap-2.5 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-xs text-rose-800">
                    <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-600" />
                    <div>
                      <p className="font-semibold">Không thể hoàn tất truy vấn</p>
                      <p className="mt-0.5 leading-relaxed">{chatError}</p>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </div>

        {/* Bottom Composer */}
        <footer className="shrink-0 border-t border-slate-200 bg-white px-5 py-3.5">
          <div className="mx-auto max-w-4xl">
            {/* Mode Switcher + External arXiv Toggle */}
            <div className="mb-2.5 flex flex-wrap items-center justify-between gap-2">
              <div className="inline-flex rounded-xl border border-slate-200 bg-slate-100 p-0.5">
                <button
                  type="button"
                  disabled={isStreaming}
                  onClick={() => setMode("agentic")}
                  className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1 text-xs font-semibold transition cursor-pointer ${
                    mode === "agentic"
                      ? "bg-white text-blue-700 shadow-2xs"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  <Sparkles className="h-3.5 w-3.5" />
                  <span>Agentic RAG</span>
                </button>
                <button
                  type="button"
                  disabled={isStreaming}
                  onClick={() => setMode("single_shot")}
                  className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1 text-xs font-semibold transition cursor-pointer ${
                    mode === "single_shot"
                      ? "bg-white text-blue-700 shadow-2xs"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  <Zap className="h-3.5 w-3.5" />
                  <span>Single-Shot RAG</span>
                </button>
              </div>

              {mode === "agentic" && (
                <label className="inline-flex items-center gap-2 text-xs text-slate-600 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={allowExternalArxiv}
                    disabled={isStreaming}
                    onChange={(e) => setAllowExternalArxiv(e.target.checked)}
                    className="h-3.5 w-3.5 rounded border-slate-300 text-blue-700 focus:ring-blue-500"
                  />
                  <Globe className="h-3.5 w-3.5 text-slate-400" />
                  <span>Tự tìm trên arXiv nếu kho local rỗng</span>
                </label>
              )}
            </div>

            {/* Input Box */}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                void sendQuery(queryInput);
              }}
              className="flex items-end gap-2.5 rounded-2xl border border-slate-300 bg-slate-50/60 p-2 focus-within:border-blue-600 focus-within:bg-white focus-within:ring-2 focus-within:ring-blue-500/15"
            >
              <textarea
                rows={2}
                value={queryInput}
                onChange={(e) => setQueryInput(e.target.value)}
                disabled={isStreaming}
                placeholder="Nhập câu hỏi nghiên cứu học thuật (Enter để gửi, Shift+Enter xuống dòng)..."
                onKeyDown={(e) => {
                  if (
                    e.key === "Enter" &&
                    !e.shiftKey &&
                    !e.nativeEvent.isComposing
                  ) {
                    e.preventDefault();
                    void sendQuery(queryInput);
                  }
                }}
                className="max-h-40 min-h-[44px] flex-1 resize-none bg-transparent px-2.5 py-1.5 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none disabled:opacity-60"
              />

              {isStreaming ? (
                <button
                  type="button"
                  onClick={handleStopStream}
                  title="Dừng luồng suy luận hiện tại"
                  className="inline-flex h-10 shrink-0 items-center gap-1.5 rounded-xl bg-rose-600 px-3.5 text-xs font-semibold text-white shadow-xs transition hover:bg-rose-700 cursor-pointer"
                >
                  <Square className="h-3.5 w-3.5 fill-current" />
                  <span>Dừng</span>
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!queryInput.trim()}
                  title="Gửi câu hỏi nghiên cứu"
                  className="inline-flex h-10 shrink-0 items-center gap-1.5 rounded-xl bg-blue-700 px-4 text-xs font-semibold text-white shadow-xs transition hover:bg-blue-800 disabled:cursor-not-allowed disabled:opacity-40 cursor-pointer"
                >
                  <span>Gửi</span>
                  <ArrowUp className="h-4 w-4" />
                </button>
              )}
            </form>

            <div className="mt-1.5 flex items-center justify-between px-1 text-[11px] text-slate-400">
              <span className="inline-flex items-center gap-1">
                <Cpu className="h-3 w-3" />
                Bấm vào nhãn trích dẫn [1], [2] trong câu trả lời để kiểm chứng
                nguyên văn.
              </span>
              <span>Enter gửi • Shift+Enter xuống dòng</span>
            </div>
          </div>
        </footer>
      </main>

      {/* Right Column: Inspector Panel */}
      <InspectorPanel
        isOpen={isInspectorOpen}
        activeTab={inspectorTab}
        onChangeTab={setInspectorTab}
        onClose={() => setIsInspectorOpen(false)}
        inspectedMessage={inspectedMessage}
        highlightedCitationIndex={highlightedCitationIndex}
        onSelectCitationIndex={(idx) => setHighlightedCitationIndex(idx)}
        liveSteps={liveSteps}
        isStreaming={isStreaming}
      />

      {/* Add Paper Modal */}
      {isAddPaperOpen && (
        <AddPaperModal
          defaultTopicId={selectedTopicId || DEFAULT_TOPIC_ID}
          onClose={() => setIsAddPaperOpen(false)}
          onIngested={handlePaperIngested}
        />
      )}
    </div>
  );
}
