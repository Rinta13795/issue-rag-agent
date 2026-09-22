import type { EvaluationSummary, ExampleItem, RunSnapshot, SystemInfo } from './types'

const API_BASE = ''

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
