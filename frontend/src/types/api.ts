export type ChatMode = "agentic" | "single_shot";

export type AgentRoute = "simple" | "decompose";

export type AgentNextAction =
  | "synthesize"
  | "refine"
  | "arxiv_discovery"
  | "abstain";

export interface ServiceStatus {
  sqlite_ok: boolean;
  qdrant_ok: boolean;
  grobid_ok: boolean;
  local_llm_ok: boolean;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  services: ServiceStatus;
  topics_count: number;
  documents_count: number;
  local_llm_model: string;
}

export interface TopicCreateRequest {
  topic_id: string;
  name: string;
  description?: string;
}

export interface TopicItem {
  id: string;
  name: string;
  description: string;
  created_at: number;
  document_count: number;
}

export interface DocumentItem {
  id: string;
  topic_id: string;
  title: string;
  source_type: string;
  source_uri: string;
  status: string;
  error_message: string;
  updated_at: number;
  paper_card: Record<string, unknown> | null;
}

export interface IngestArxivRequest {
  arxiv_id: string;
  topic_id: string;
  force?: boolean;
}

export interface IngestResponse {
  doc_id: string;
  topic_id: string;
  status: string;
  chunks_count: number;
  note: string;
}

export interface SessionCreateRequest {
  topic_id?: string | null;
  title?: string;
}

export interface SessionUpdateRequest {
  title: string;
}

export interface SessionResponse {
  session_id: string;
  topic_id: string | null;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface Citation {
  index: number;
  doc_id: string;
  title: string;
  section: string;
  chunk_index: number;
  snippet: string;
}

export interface SubQuery {
  sub_query_id: string;
  search_query: string;
  bm25_keywords: string[];
  target_aspect: string;
}

export interface QueryPlan {
  route: AgentRoute;
  sub_queries: SubQuery[];
  reasoning: string;
}

export interface EvidenceGrade {
  is_sufficient: boolean;
  missing_aspects: string[];
  follow_up_queries: SubQuery[];
  rationale: string;
}

export interface AgentStepTrace {
  node: string;
  hop: number;
  latency_ms: number;
  detail: string;
}

export interface AgentExecutionTrace {
  route: AgentRoute;
  hops_executed: number;
  used_fast_path: boolean;
  used_external_arxiv: boolean;
  llm_calls: number;
  plan: QueryPlan | null;
  grades: EvidenceGrade[];
  steps: AgentStepTrace[];
}

export interface MessageResponse {
  message_id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  agent_trace: AgentExecutionTrace | null;
  is_grounded: boolean | null;
  abstain_reason: string;
  created_at: string;
}

export interface ChatRequest {
  query: string;
  session_id?: string | null;
  topic_id?: string | null;
  doc_ids?: string[] | null;
  mode?: ChatMode;
  allow_external_arxiv?: boolean;
  max_hops?: number | null;
  top_k?: number | null;
}

export interface ChatResponse {
  session_id: string;
  user_message_id: string;
  assistant_message_id: string;
  mode: ChatMode;
  query: string;
  answer: string;
  citations: Citation[];
  is_grounded: boolean;
  abstain_reason: string;
  invalid_citations_removed: number[];
  agent_trace: AgentExecutionTrace | null;
}

export interface StreamSessionEvent {
  session_id: string;
  user_message_id: string;
  mode: ChatMode;
}

export interface StreamErrorEvent {
  detail: string;
}
