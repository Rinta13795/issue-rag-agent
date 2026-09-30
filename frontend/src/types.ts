export type RunStatus = 'queued' | 'running' | 'completed' | 'failed'
export type NodeStatus = 'waiting' | 'running' | 'completed' | 'failed'

export interface QueryAnalysisInfo {
  rewritten_query: string
  keywords: string[]
  component: string | null
  component_filter_applied: boolean
  component_filter_note: string | null
}

export interface CandidateInfo {
  id: string
  title: string
  body_snippet: string
  score?: number
  rerank_score?: number
  is_related: boolean
}

export interface DecisionInfo {
  decision: string
  confidence: number
  related_issues: string[]
  reasoning: string
  retry_count: number
}

export interface RetryRoundInfo {
  round: number
  rewritten_query: string
  keywords: string[]
  component: string | null
  decision: string
  confidence: number
  related_issues: string[]
  reasoning: string
  retrieved_count: number
  candidate_count: number
  top_score?: number
  score_gap?: number
}

export interface PipelineNodeInfo {
  name: string
  label: string
  status: NodeStatus
  elapsed_ms?: number
  summary?: string
}

export interface RunSnapshot {
  run_id: string
  status: RunStatus
  created_at: string
  started_at?: string
  completed_at?: string
  total_elapsed_ms?: number
  current_node?: string
  current_round: number
  issue_text: string
  nodes: PipelineNodeInfo[]
  analysis?: QueryAnalysisInfo
  retrieval?: {
    candidate_count: number
    candidates: CandidateInfo[]
  }
  rerank?: {
    candidate_count: number
    candidates: CandidateInfo[]
  }
  decision?: DecisionInfo
  rounds: RetryRoundInfo[]
  node_timings_ms: Record<string, number>
  error?: string
}

export interface ExampleItem {
  id: string
  label: string
  tag: string
  title: string
  description: string
  expected_decision: string
  note: string
}

export interface SystemInfo {
  version: string
  dataset_size: number
  dataset_name: string
  embedding_model: string
  reranker_model: string
  llm_model: string
  bm25_ready: boolean
  vector_ready: boolean
  docstore_ready: boolean
  local_demo: boolean
  known_limitations: string[]
}

export interface EvaluationSummary {
  sample_size: number
  total_test_queries: number
  seed: number
  self_hit_excluded: boolean
  metrics: Record<string, Record<string, number>>
  notes: string[]
}

export type ChatStatus = 'idle' | 'thinking' | 'retrieving' | 'answering' | 'completed' | 'failed' | 'waiting_for_user'

export interface ChatFact {
  value: string
  source_message_id: string
  source_excerpt: string
}

export interface ChatCandidate {
  id: string
  title: string
  body_snippet: string
  rerank_score: number | null
  source: 'local' | 'github'
  url: string | null
}

export interface SourceIssue {
  repository: string
  number: number
  title: string
  body: string
  state: string
  url: string
  comments: { author: string; body: string; url: string }[]
  comments_truncated: boolean
}

export interface IssueDraft {
  draft_id: string
  title: string
  body: string
  version: number
  status: 'draft' | 'publishing' | 'published' | 'uncertain'
  published_url: string | null
  published_number: number | null
  possible_duplicates: ChatCandidate[]
  search_status: 'not_run' | 'ok' | 'failed'
}

export interface MemoryProposal {
  kind: 'preference' | 'experience'
  text: string
  scope: 'global' | 'repository'
  source_excerpt: string
}

export interface MemoryRecord extends MemoryProposal {
  memory_id: string
  repository_id: string | null
  source_session_id: string
  created_at: string
  status: 'observed' | 'active' | 'pending' | 'supported' | 'verified' | 'refuted'
  origin: 'explicit' | 'inferred' | 'user' | 'assistant' | 'legacy'
  entry_type: 'preference' | 'observation' | 'plan' | 'hypothesis' | 'attempt' | 'result' | 'summary'
  case_id: string | null
  source_refs: MemorySourceRef[]
  supporting_sessions: string[]
  revision: number
  revision_history: MemoryRevision[]
  modified_by: 'automatic' | 'user'
}

export interface MemorySourceRef {
  source_type: 'user_message' | 'source_issue' | 'candidate' | 'published_issue' | 'tool_evidence'
  source_id: string
  excerpt: string
  url: string | null
  created_at: string
}

export interface MemoryRevision {
  revision: number
  text: string
  status: string
  changed_at: string
  changed_by: 'automatic' | 'user'
  reason: string | null
}

export interface MemoryCase {
  case_id: string
  repository_id: string
  title: string
  summary: string
  source_issue_url: string | null
  source_issue_id: string | null
  session_ids: string[]
  updated_at: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
  citations: string[]
  action: string | null
}

export interface ChatSession {
  session_id: string
  repository_id: string | null
  source_issue: SourceIssue | null
  pending_question: { question_id: string; call_id: string; question: string; options: string[]; turn_id: string } | null
  runtime_steps: { call_id: string; turn_id: string; tool: string; status: string; arguments: Record<string, unknown>; result: Record<string, unknown> }[]
  evidence: { id: string; kind: string; title: string; text: string; url: string | null; fetched_at: string; truncated: boolean }[]
  issue_draft: IssueDraft | null
  memory_proposal: MemoryProposal | null
  memory_case_id: string | null
  investigation_summary: string
  memory_organization: { enabled: boolean; status: 'idle' | 'queued' | 'processing' | 'completed' | 'failed'; last_error: string | null }
  selected_memory_context: { preferences: Array<Record<string, unknown>>; cases: Array<Record<string, unknown>> }
  created_at: string
  updated_at: string
  status: ChatStatus
  messages: ChatMessage[]
  facts: ChatFact[]
  open_question: string | null
  focus_candidate_id: string | null
  candidates: ChatCandidate[]
  last_search_fingerprint: string | null
  live_search_status: 'not_run' | 'ok' | 'failed'
  live_search_message: string | null
  model_calls: number
  retrieval_calls: number
  prompt_tokens: number | null
  completion_tokens: number | null
  last_elapsed_ms: number | null
  last_error: string | null
}

export interface ChatSessionSummary {
  session_id: string
  repository_id: string | null
  title: string
  updated_at: string
  status: ChatStatus
}

export interface ChatRepository {
  id: string
  label: string
  issue_count: number
  source: string
  github_url: string | null
}

export interface SyncRepositoryStatus {
  job_id: string
  repository: string
  status: 'queued' | 'fetching' | 'indexing' | 'completed' | 'failed'
  message: string
  repository_id: string | null
}
