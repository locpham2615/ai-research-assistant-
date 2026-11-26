import type {
  AgentStepTrace,
  ChatRequest,
  ChatResponse,
  DocumentItem,
  HealthResponse,
  IngestArxivRequest,
  IngestResponse,
  MessageResponse,
  SessionCreateRequest,
  SessionResponse,
  SessionUpdateRequest,
  StreamErrorEvent,
  StreamSessionEvent,
  TopicCreateRequest,
  TopicItem,
} from "@/types/api";

export const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
).replace(/\/+$/, "");

export class ApiError extends Error {
  public readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

interface FastApiErrorShape {
  detail?: string | Array<{ msg?: string; loc?: Array<string | number> }>;
}

async function parseErrorMessage(res: Response): Promise<string> {
  try {
    const data = (await res.json()) as FastApiErrorShape;
    if (typeof data.detail === "string" && data.detail.trim()) {
      return data.detail;
    }
    if (Array.isArray(data.detail) && data.detail.length > 0) {
      return data.detail
        .map((item) => {
          const loc = Array.isArray(item.loc) ? item.loc.join(".") : "";
          const msg = item.msg || "Dữ liệu không hợp lệ";
          return loc ? `${loc}: ${msg}` : msg;
        })
        .join("; ");
    }
  } catch {
    // Fallback below if body is not valid JSON
  }
  return `Lỗi máy chủ (${res.status} ${res.statusText || "HTTP Error"})`;
}

async function requestJson<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.headers ?? {}),
      },
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw err;
    }
    throw new ApiError(
      `Không thể kết nối tới FastAPI backend (${API_BASE_URL}). Vui lòng kiểm tra server đã bật chưa.`,
      0
    );
  }

  if (!res.ok) {
    const msg = await parseErrorMessage(res);
    throw new ApiError(msg, res.status);
  }

  if (res.status === 204) {
    return undefined as T;
  }

  return (await res.json()) as T;
}

export async function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/api/health", {
    method: "GET",
    cache: "no-store",
  });
}

export async function listTopics(): Promise<TopicItem[]> {
  return requestJson<TopicItem[]>("/api/topics", {
    method: "GET",
    cache: "no-store",
  });
}

export async function createTopic(
  payload: TopicCreateRequest
): Promise<TopicItem> {
  return requestJson<TopicItem>("/api/topics", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function listDocuments(topicId?: string): Promise<DocumentItem[]> {
  const query = topicId
    ? `?topic_id=${encodeURIComponent(topicId)}`
    : "";
  return requestJson<DocumentItem[]>(`/api/documents${query}`, {
    method: "GET",
    cache: "no-store",
  });
}

export async function ingestArxiv(
  payload: IngestArxivRequest
): Promise<IngestResponse> {
  return requestJson<IngestResponse>("/api/documents/ingest-arxiv", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      arxiv_id: payload.arxiv_id,
      topic_id: payload.topic_id,
      force: Boolean(payload.force),
    }),
  });
}

export async function uploadPdfDocument(params: {
  file: File;
  topic_id: string;
  force?: boolean;
}): Promise<IngestResponse> {
  const formData = new FormData();
  formData.append("file", params.file);
  formData.append("topic_id", params.topic_id);
  formData.append("force", params.force ? "true" : "false");

  return requestJson<IngestResponse>("/api/documents/upload", {
    method: "POST",
    body: formData,
  });
}

export async function listSessions(params?: {
  topic_id?: string;
  limit?: number;
}): Promise<SessionResponse[]> {
  const search = new URLSearchParams();
  if (params?.topic_id) {
    search.set("topic_id", params.topic_id);
  }
  if (typeof params?.limit === "number") {
    search.set("limit", String(params.limit));
  }
  const qs = search.toString();
  return requestJson<SessionResponse[]>(
    `/api/sessions${qs ? `?${qs}` : ""}`,
    {
      method: "GET",
      cache: "no-store",
    }
  );
}

export async function createSession(
  payload: SessionCreateRequest
): Promise<SessionResponse> {
  return requestJson<SessionResponse>("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function renameSession(
  sessionId: string,
  payload: SessionUpdateRequest
): Promise<SessionResponse> {
  return requestJson<SessionResponse>(
    `/api/sessions/${encodeURIComponent(sessionId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }
  );
}

export async function deleteSession(sessionId: string): Promise<void> {
  return requestJson<void>(
    `/api/sessions/${encodeURIComponent(sessionId)}`,
    {
      method: "DELETE",
    }
  );
}

export async function getSessionMessages(
  sessionId: string
): Promise<MessageResponse[]> {
  return requestJson<MessageResponse[]>(
    `/api/sessions/${encodeURIComponent(sessionId)}/messages`,
    {
      method: "GET",
      cache: "no-store",
    }
  );
}

export async function chatOnce(payload: ChatRequest): Promise<ChatResponse> {
  return requestJson<ChatResponse>("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export interface StreamChatCallbacks {
  onSession?: (event: StreamSessionEvent) => void;
  onStep?: (step: AgentStepTrace) => void;
  onFinal?: (response: ChatResponse) => void;
  onError?: (errorMessage: string) => void;
}

interface ParsedSseEvent {
  event: string;
  data: string;
}

function parseSseBlock(rawBlock: string): ParsedSseEvent | null {
  const lines = rawBlock.split("\n");
  let eventName = "message";
  const dataLines: string[] = [];

  for (const line of lines) {
    if (!line || line.startsWith(":")) {
      continue;
    }
    if (line.startsWith("event:")) {
      const value = line.slice("event:".length);
      eventName = (value.startsWith(" ") ? value.slice(1) : value).trim();
    } else if (line.startsWith("data:")) {
      const value = line.slice("data:".length);
      dataLines.push(value.startsWith(" ") ? value.slice(1) : value);
    }
  }

  if (dataLines.length === 0) {
    return null;
  }

  return {
    event: eventName,
    data: dataLines.join("\n"),
  };
}

function dispatchSseEvent(
  parsed: ParsedSseEvent,
  callbacks: StreamChatCallbacks
): void {
  switch (parsed.event) {
    case "session": {
      const payload = JSON.parse(parsed.data) as StreamSessionEvent;
      callbacks.onSession?.(payload);
      break;
    }
    case "step": {
      const payload = JSON.parse(parsed.data) as AgentStepTrace;
      callbacks.onStep?.(payload);
      break;
    }
    case "final": {
      const payload = JSON.parse(parsed.data) as ChatResponse;
      callbacks.onFinal?.(payload);
      break;
    }
    case "error": {
      let detail = parsed.data;
      try {
        const errPayload = JSON.parse(parsed.data) as StreamErrorEvent;
        if (errPayload.detail) {
          detail = errPayload.detail;
        }
      } catch {
        // Use raw text if not JSON
      }
      callbacks.onError?.(detail);
      break;
    }
    default:
      break;
  }
}

export async function streamChat(
  payload: ChatRequest,
  callbacks: StreamChatCallbacks,
  signal?: AbortSignal
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/api/chat/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw err;
    }
    throw new ApiError(
      `Không thể kết nối tới FastAPI backend (${API_BASE_URL}). Vui lòng kiểm tra server đã bật chưa.`,
      0
    );
  }

  if (!res.ok) {
    const msg = await parseErrorMessage(res);
    throw new ApiError(msg, res.status);
  }

  if (!res.body) {
    throw new ApiError("Trình duyệt không hỗ trợ đọc luồng phản hồi (ReadableStream).", 500);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        buffer += decoder.decode();
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n");

      let separatorIndex = buffer.indexOf("\n\n");
      while (separatorIndex !== -1) {
        const rawBlock = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + 2);

        const parsed = parseSseBlock(rawBlock);
        if (parsed) {
          dispatchSseEvent(parsed, callbacks);
        }

        separatorIndex = buffer.indexOf("\n\n");
      }
    }

    buffer = buffer.replace(/\r\n/g, "\n").trim();
    if (buffer) {
      const parsed = parseSseBlock(buffer);
      if (parsed) {
        dispatchSseEvent(parsed, callbacks);
      }
    }
  } finally {
    reader.releaseLock();
  }
}
