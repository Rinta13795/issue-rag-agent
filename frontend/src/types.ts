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
