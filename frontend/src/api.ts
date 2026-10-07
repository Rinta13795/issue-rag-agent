import type { ChatRepository, ChatSession, ChatSessionSummary, MemoryCase, MemoryRecord, SourceIssue, SyncRepositoryStatus, EvaluationSummary, ExampleItem, RunSnapshot, SystemInfo } from './types'

const API_BASE = ''

export class ChatApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function chatJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new ChatApiError(err.detail || `请求失败 (${res.status})`, res.status)
  }
  return res.json()
}

export async function listChatRepositories(): Promise<ChatRepository[]> {
  return chatJson(await fetch(`${API_BASE}/api/chat/repositories`))
}

export async function syncChatRepository(repository: string): Promise<SyncRepositoryStatus> {
  return chatJson(await fetch(`${API_BASE}/api/chat/repositories/sync`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ repository }),
  }))
}

export async function getRepositorySync(jobId: string): Promise<SyncRepositoryStatus> {
  return chatJson(await fetch(`${API_BASE}/api/chat/repositories/sync/${encodeURIComponent(jobId)}`))
}

export async function listChatSessions(): Promise<ChatSessionSummary[]> {
  return chatJson(await fetch(`${API_BASE}/api/chat/sessions`))
}

export async function previewChatIssue(issueUrl: string): Promise<SourceIssue> {
  return chatJson(await fetch(`${API_BASE}/api/chat/issues/preview`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ issue_url: issueUrl }),
  }))
}

export async function createChatSession(repositoryId: string, issueUrl?: string, caseId?: string): Promise<ChatSession> {
  return chatJson(await fetch(`${API_BASE}/api/chat/sessions`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ repository_id: repositoryId, issue_url: issueUrl || null, case_id: caseId || null }),
  }))
}

export async function getChatSession(sessionId: string): Promise<ChatSession> {
  return chatJson(await fetch(`${API_BASE}/api/chat/sessions/${encodeURIComponent(sessionId)}`))
}

export function listenChatEvents(sessionId: string, handlers: {
  snapshot: (session: ChatSession) => void
  answer: (answer: string, turnId: string | null) => void
  connection: (connected: boolean) => void
  expired: () => void
}): () => void {
  const stream = new EventSource(`${API_BASE}/api/chat/sessions/${encodeURIComponent(sessionId)}/events`)
  stream.addEventListener('snapshot', (event) => handlers.snapshot(JSON.parse((event as MessageEvent).data)))
  stream.addEventListener('answer', (event) => {
    const value = JSON.parse((event as MessageEvent).data)
    handlers.answer(value.answer, value.turn_id)
  })
  stream.addEventListener('expired', () => { stream.close(); handlers.expired() })
  stream.onopen = () => handlers.connection(true)
  stream.onerror = () => handlers.connection(false)
  return () => stream.close()
}

export async function sendChatMessage(sessionId: string, content: string, clientMessageId: string): Promise<void> {
  await chatJson(await fetch(`${API_BASE}/api/chat/sessions/${encodeURIComponent(sessionId)}/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content, client_message_id: clientMessageId }),
  }))
}

export async function retryChatMessage(sessionId: string): Promise<void> {
  await chatJson(await fetch(`${API_BASE}/api/chat/sessions/${encodeURIComponent(sessionId)}/retry`, { method: 'POST' }))
}

export async function createIssueDraft(sessionId: string): Promise<ChatSession> {
  return chatJson(await fetch(`${API_BASE}/api/chat/sessions/${encodeURIComponent(sessionId)}/issue-draft`, { method: 'POST' }))
}

export async function editIssueDraft(sessionId: string, title: string, body: string, version: number): Promise<ChatSession> {
  return chatJson(await fetch(`${API_BASE}/api/chat/sessions/${encodeURIComponent(sessionId)}/issue-draft`, {
    method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title, body, version }),
  }))
}

export async function publishIssueDraft(sessionId: string, draftId: string, version: number): Promise<ChatSession> {
  return chatJson(await fetch(`${API_BASE}/api/chat/sessions/${encodeURIComponent(sessionId)}/issue-draft/publish`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ draft_id: draftId, version, confirmed: true }),
  }))
}

export async function proposeMemory(sessionId: string): Promise<ChatSession> {
  return chatJson(await fetch(`${API_BASE}/api/chat/sessions/${encodeURIComponent(sessionId)}/memory-proposal`, { method: 'POST' }))
}

export async function confirmMemory(sessionId: string, text: string, kind: 'preference' | 'experience', scope: 'global' | 'repository'): Promise<MemoryRecord> {
  return chatJson(await fetch(`${API_BASE}/api/chat/sessions/${encodeURIComponent(sessionId)}/memory`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, kind, scope, confirmed: true }),
  }))
}

export async function listMemories(repositoryId?: string): Promise<MemoryRecord[]> {
  const query = repositoryId ? `?repository_id=${encodeURIComponent(repositoryId)}` : ''
  return chatJson(await fetch(`${API_BASE}/api/chat/memories${query}`))
}

export async function deleteMemory(memoryId: string): Promise<void> {
  await chatJson(await fetch(`${API_BASE}/api/chat/memories/${encodeURIComponent(memoryId)}`, { method: 'DELETE' }))
}

export async function updateMemory(memoryId: string, text: string, status: MemoryRecord['status']): Promise<MemoryRecord> {
  return chatJson(await fetch(`${API_BASE}/api/chat/memories/${encodeURIComponent(memoryId)}`, {
    method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, status }),
  }))
}

export async function listMemoryCases(repositoryId: string): Promise<MemoryCase[]> {
  return chatJson(await fetch(`${API_BASE}/api/chat/memory-cases?repository_id=${encodeURIComponent(repositoryId)}`))
}

export async function getMemorySettings(): Promise<{ auto_capture: boolean }> {
  return chatJson(await fetch(`${API_BASE}/api/chat/memory-settings`))
}

export async function updateMemorySettings(autoCapture: boolean): Promise<{ auto_capture: boolean }> {
  return chatJson(await fetch(`${API_BASE}/api/chat/memory-settings`, {
    method: 'PATCH', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ auto_capture: autoCapture }),
  }))
}

export async function createRun(issueText: string, sampleId?: string): Promise<{ run_id: string; status: string; created_at: string }> {
  const res = await fetch(`${API_BASE}/api/runs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ issue_text: issueText, sample_id: sampleId || null }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `请求失败 (${res.status})`)
  }
  return res.json()
}

export async function getRunSnapshot(runId: string): Promise<RunSnapshot> {
  const res = await fetch(`${API_BASE}/api/runs/${runId}`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `获取快照失败 (${res.status})`)
  }
  return res.json()
}

export function listenRunEvents(
  runId: string,
  onSnapshot: (snapshot: RunSnapshot) => void,
  onEvent: (eventType: string, data: any) => void,
  onFinish: () => void,
  onError: (err: any) => void
): () => void {
  const eventSource = new EventSource(`${API_BASE}/api/runs/${runId}/events`)

  eventSource.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data)
      const eventType = payload.event
      const data = payload.data

      if (eventType === 'run.snapshot') {
        onSnapshot(data)
      } else {
        onEvent(eventType, data)
        // 节点或重试事件发生后，主动拉取一次最新完整快照保证数据一致
        getRunSnapshot(runId).then(onSnapshot).catch(console.error)
      }

      if (eventType === 'run.completed' || eventType === 'run.failed') {
        getRunSnapshot(runId).then((latest) => {
          onSnapshot(latest)
          onFinish()
          eventSource.close()
        }).catch(() => {
          onFinish()
          eventSource.close()
        })
      }
    } catch (e) {
      console.error('Failed to parse SSE event:', e)
    }
  }

  eventSource.onerror = (err) => {
    console.warn('SSE stream closed or error, falling back to final polling', err)
    getRunSnapshot(runId).then((snapshot) => {
      onSnapshot(snapshot)
      if (snapshot.status === 'completed' || snapshot.status === 'failed') {
        onFinish()
      }
    }).catch(onError)
    eventSource.close()
  }

  return () => {
    eventSource.close()
  }
}

export async function fetchExamples(): Promise<ExampleItem[]> {
  const res = await fetch(`${API_BASE}/api/examples`)
  if (!res.ok) throw new Error('获取示例失败')
  return res.json()
}

export async function fetchSystemInfo(): Promise<SystemInfo> {
  const res = await fetch(`${API_BASE}/api/system`)
  if (!res.ok) throw new Error('获取系统信息失败')
  return res.json()
}

export async function fetchEvaluationSummary(): Promise<EvaluationSummary> {
  const res = await fetch(`${API_BASE}/api/evaluation/summary`)
  if (!res.ok) throw new Error('获取评测数据失败')
  return res.json()
}

export async function answerChatQuestion(sessionId: string, questionId: string, answer: string, clientMessageId: string, cancelled = false): Promise<{ message_id: string; session_id: string; status: string }> {
  return chatJson(await fetch(`${API_BASE}/api/chat/sessions/${encodeURIComponent(sessionId)}/answers`, {
    method: 'POST', body: JSON.stringify({ question_id: questionId, answer, client_message_id: clientMessageId, cancelled }),
    headers: { 'Content-Type': 'application/json' },
  }))
}
