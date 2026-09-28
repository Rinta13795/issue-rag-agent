import { useCallback, useEffect, useRef, useState } from 'react'
import { ChatApiError, createChatSession, getChatSession, listChatRepositories, listChatSessions, sendChatMessage } from '../api'
import type { ChatRepository, ChatSession, ChatSessionSummary } from '../types'

const STORAGE_KEY = 'issue-rag-local-chat-session'
const activeStates = new Set(['thinking', 'retrieving', 'answering'])
const statusLabels: Record<string, string> = {
  thinking: '正在理解这轮对话',
  retrieving: '正在查找历史 Issue',
  answering: '正在整理证据与回答',
}

function issueUrl(id: string): string | null {
  const match = /^openharness:(\d+)$/.exec(id)
  return match ? `https://github.com/HKUDS/OpenHarness/issues/${match[1]}` : null
}

interface Props {
  onOpenTriage: () => void
}

export function ChatWorkspace({ onOpenTriage }: Props) {
  const [session, setSession] = useState<ChatSession | null>(null)
  const [sessions, setSessions] = useState<ChatSessionSummary[]>([])
  const [repositories, setRepositories] = useState<ChatRepository[]>([])
  const [selectedRepository, setSelectedRepository] = useState('')
  const [draft, setDraft] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [isSending, setIsSending] = useState(false)
  const loadedRef = useRef(false)
  const selectionEpochRef = useRef(0)
  const endRef = useRef<HTMLDivElement | null>(null)
  const sessionId = session?.session_id
  const sessionStatus = session?.status

  const refreshSessions = useCallback(async () => setSessions(await listChatSessions()), [])

  const forgetMissingSession = useCallback((missingId: string) => {
    if (window.localStorage.getItem(STORAGE_KEY) !== missingId) return
    selectionEpochRef.current += 1
    window.localStorage.removeItem(STORAGE_KEY)
    setSession(null)
    setNotice('旧会话已过期。请选择仓库开始新对话。')
    void refreshSessions()
  }, [refreshSessions])

  useEffect(() => {
    if (loadedRef.current) return
    loadedRef.current = true
    const savedId = window.localStorage.getItem(STORAGE_KEY)
    const loadEpoch = selectionEpochRef.current
    const load = async () => {
      try {
        const [available, history] = await Promise.all([listChatRepositories(), listChatSessions()])
        setRepositories(available)
        setSessions(history)
        if (savedId) {
          const restored = await getChatSession(savedId)
          if (selectionEpochRef.current === loadEpoch && window.localStorage.getItem(STORAGE_KEY) === savedId) setSession(restored)
        }
      } catch (caught) {
        if (savedId && caught instanceof ChatApiError && caught.status === 404) {
          forgetMissingSession(savedId)
        } else {
          setError('暂时无法连接对话服务，请确认后端已经启动。')
        }
      }
    }
    void load()
  }, [forgetMissingSession])

  useEffect(() => {
    if (!sessionId || !sessionStatus || !activeStates.has(sessionStatus)) return
    const timer = window.setInterval(() => {
      getChatSession(sessionId).then((updated) => {
        if (window.localStorage.getItem(STORAGE_KEY) === sessionId) setSession(updated)
      }).catch((caught) => {
        if (caught instanceof ChatApiError && caught.status === 404) {
          forgetMissingSession(sessionId)
        } else {
          setError('会话连接中断，请稍后重试。')
        }
      })
    }, 900)
    return () => window.clearInterval(timer)
  }, [sessionId, sessionStatus, forgetMissingSession])

  useEffect(() => { if (sessionStatus === 'completed' || sessionStatus === 'failed') void refreshSessions() }, [sessionStatus, refreshSessions])

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [session?.messages.length])

  const startNew = () => {
    selectionEpochRef.current += 1
    window.localStorage.removeItem(STORAGE_KEY)
    setSession(null)
    setSelectedRepository('')
    setDraft('')
    setError(null)
    setNotice(null)
  }

  const createForRepository = async () => {
    if (!selectedRepository) return
    const createEpoch = ++selectionEpochRef.current
    try {
      const created = await createChatSession(selectedRepository)
      if (selectionEpochRef.current !== createEpoch) return
      window.localStorage.setItem(STORAGE_KEY, created.session_id)
      setSession(created)
      setError(null)
      setNotice(null)
      await refreshSessions()
    } catch {
      setError('暂时无法创建会话，请确认后端已经启动。')
    }
  }

  const openSession = async (id: string) => {
    const openEpoch = ++selectionEpochRef.current
    try {
      const opened = await getChatSession(id)
      if (selectionEpochRef.current !== openEpoch) return
      window.localStorage.setItem(STORAGE_KEY, id)
      setSession(opened)
      setDraft('')
      setNotice(null)
      setError(null)
    } catch {
      if (window.localStorage.getItem(STORAGE_KEY) === id) forgetMissingSession(id)
    }
  }

  const send = async (suggestion?: string) => {
    const content = (suggestion ?? draft).trim()
    if (!session || activeStates.has(session.status) || isSending || !content || content === '报错原文：') return
    setIsSending(true)
    setError(null)
    setNotice(null)
    try {
      await sendChatMessage(session.session_id, content, window.crypto.randomUUID())
      if (!suggestion) setDraft('')
      const updated = await getChatSession(session.session_id)
      if (window.localStorage.getItem(STORAGE_KEY) === session.session_id) setSession(updated)
    } catch (caught) {
      if (caught instanceof ChatApiError && caught.status === 404) {
        forgetMissingSession(session.session_id)
      } else {
        setError(caught instanceof Error ? caught.message : '消息发送失败')
      }
    } finally {
      setIsSending(false)
    }
  }

  const busy = isSending || Boolean(session && activeStates.has(session.status))
  const activeRepository = repositories.find((repo) => repo.id === session?.repository_id)
  const canSendDraft = Boolean(draft.trim() && draft.trim() !== '报错原文：')
  const hasConversation = Boolean(session?.messages.length) || busy
  const canSearchHistory = Boolean(session?.messages.some((message) =>
    message.role === 'user' && /error|exception|traceback|fail|crash|报错|失败|崩溃|卡住|无法|不能|不了/i.test(message.content),
  ))
  const prefill = (text: string) => {
    setDraft(text)
    document.getElementById('chat-input')?.focus()
  }

  const composer = <div className="chat-composer">
    <label className="field-label" htmlFor="chat-input">{hasConversation ? '继续对话' : '描述你的问题'}</label>
    <textarea
      id="chat-input" className="field-textarea" value={draft} maxLength={12000}
      placeholder={hasConversation ? '补充线索，或追问：第一条为什么像？' : '例如：升级依赖后启动失败，贴上报错或说说你观察到的现象…'}
      onChange={(event) => setDraft(event.target.value)}
      onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void send() } }}
    />
    <div className="chat-composer-foot"><span>Enter 发送 · Shift + Enter 换行</span><button type="button" className="btn btn--primary" disabled={!session || busy || !canSendDraft} onClick={() => void send()}>发送消息</button></div>
  </div>

  return <main className={`chat-workspace chat-workspace--scoped ${hasConversation ? 'chat-workspace--thread' : 'chat-workspace--home'}`}>
    <aside className="chat-session-list" aria-label="历史对话">
      <button type="button" className="btn btn--sm" onClick={startNew}>＋ 新对话</button>
      <p className="chat-session-heading">最近对话</p>
      {sessions.map((item) => <button type="button" key={item.session_id}
        className={`chat-session-item ${session?.session_id === item.session_id ? 'chat-session-item--active' : ''}`}
        onClick={() => void openSession(item.session_id)}>
        <span>{item.title}</span><small>{item.repository_id || '旧会话 · 未绑定仓库'}</small>
      </button>)}
      {!sessions.length && <p className="chat-side-note">还没有对话。</p>}
    </aside>
    <div className="chat-conversation-area">
    {!hasConversation && <section className="chat-home" aria-label="开始对话">
      <div className="masthead-pre">ISSUE TRIAGE / CONVERSATION</div>
      <h1 className="masthead-title">你遇到了什么问题<span className="dot">？</span></h1>
      <p className="masthead-sub">先选一个仓库，再描述报错或故障现象。这个对话只会检索该仓库已同步的 Issue，不读取你的终端或实时 GitHub。</p>
      {!session && <div className="chat-repository-picker">
        <label className="field-label" htmlFor="chat-repository">这次要查哪个仓库？</label>
        <select id="chat-repository" value={selectedRepository} onChange={(event) => setSelectedRepository(event.target.value)}>
          <option value="">请选择已索引的仓库</option>
          {repositories.map((repo) => <option key={repo.id} value={repo.id}>{repo.label} · {repo.issue_count.toLocaleString()} 条 Issue</option>)}
        </select>
        {selectedRepository && <p className="chat-side-note">{repositories.find((repo) => repo.id === selectedRepository)?.source}；仓库选定后，本对话中不可切换。</p>}
        <button type="button" className="btn btn--primary" disabled={!selectedRepository} onClick={() => void createForRepository()}>开始对话</button>
      </div>}
      {notice && <p className="chat-connection" role="status">{notice}</p>}
      {error && <div className="chat-connection-error" role="alert"><p className="chat-error">■ {error}</p><button type="button" className="btn btn--sm" onClick={startNew}>重新选择</button></div>}
      {session && <p className="chat-connection">当前仓库：{activeRepository?.label || '旧会话'} · 仅检索此仓库本地快照</p>}
      {session && composer}
      <div className="chat-home-foot">
        <span>不确定怎么描述？从一句话开始也可以。</span>
        <button type="button" onClick={onOpenTriage}>使用单次分诊 →</button>
      </div>
    </section>}

    {hasConversation && <section className="chat-thread" aria-label="对话">
      <div className="chat-section-head">
        <div className="block-label"><span className="cn">{activeRepository?.label || '旧会话'}</span><span className="en">CONVERSATION · {session?.repository_id || 'UNSCOPED'}</span></div>
        <button type="button" className="btn btn--sm" onClick={startNew}>新对话</button>
      </div>
      <div className="chat-thread-content">
        <div className="chat-messages" aria-live="polite">
          {session?.messages.map((message, index) => <article className={`chat-message chat-message--${message.role}`} key={message.id}>
            <div className="chat-message-meta">{message.role === 'user' ? '你' : 'Issue Agent'} · {message.action === 'retrieve' ? '已检索证据' : message.action === 'clarify' ? '需要澄清' : message.action === 'reply' ? '基于当前上下文' : '对话'}</div>
            <p>{message.content}</p>
            {message.citations.length > 0 && <div className="chat-citation">依据：{message.citations.map((id, index) => <span key={id}>{index > 0 ? ' · ' : ''}{issueUrl(id) ? <a href={issueUrl(id)!} target="_blank" rel="noreferrer">{id} ↗</a> : id}</span>)}</div>}
            {message.action === 'clarify' && index === session.messages.length - 1 && !busy && <div className="chat-clarify-options" aria-label="下一步选择">
              {session.retrieval_calls === 0 && canSearchHistory && <button type="button" className="btn btn--sm" onClick={() => void send('请先用我已经描述的故障现象搜索本地历史 Issue。')}>先查历史 Issue</button>}
              {session.retrieval_calls > 0 ? <>
                <button type="button" className="btn btn--sm" onClick={() => prefill('报错原文：\n')}>补充报错原文</button>
                <button type="button" className="btn btn--sm" onClick={() => setNotice('目前只查本地已索引的历史 Issue；查不到不代表其他仓库或最新 Issue 没有解法。')}>查看检索范围</button>
              </> : <button type="button" className="btn btn--sm" onClick={() => prefill(draft)}>补充线索</button>}
            </div>}
          </article>)}
          {busy && <div className="chat-progress" role="status">{statusLabels[session?.status || ''] || '正在发送'}…</div>}
          <div ref={endRef} />
        </div>
        {(error || session?.last_error) && <p className="chat-error">■ {error || session?.last_error}</p>}
        {notice && <p className="chat-connection" role="status">{notice}</p>}
        {session?.repository_id ? composer : <p className="chat-side-note">这是升级前未绑定仓库的旧对话。请点“新对话”并选择仓库后继续。</p>}
        <details className="chat-evidence">
          <summary>查看已知线索与历史候选 <span>{session?.facts.length || 0} 条线索 · {session?.candidates.length || 0} 条候选</span></summary>
          <div className="chat-evidence-content">
        <section className="block">
          <div className="block-label"><span className="cn">已知线索</span><span className="en">USER FACTS</span></div>
          {session?.facts.length ? <ul className="chat-facts">{session.facts.map((fact, index) => <li key={`${fact.source_message_id}-${index}`}>{fact.value}</li>)}</ul> : <p className="chat-side-note">只记录你明确说出的故障线索，不把模型猜测写入问题卡片。</p>}
          {session?.open_question && <p className="chat-open-question">还需确认：{session.open_question}</p>}
        </section>
        <section className="block">
          <div className="block-label"><span className="cn">历史候选</span><span className="en">ISSUE EVIDENCE</span></div>
          {session?.candidates.length ? <ol className="chat-candidates">{session.candidates.map((candidate) => <li key={candidate.id}>
            <span className="chat-candidate-id">{issueUrl(candidate.id) ? <a href={issueUrl(candidate.id)!} target="_blank" rel="noreferrer">{candidate.id} ↗ 查看原文</a> : candidate.id}</span>
            <b>{candidate.title || '无标题 Issue'}</b>
            <p>{candidate.body_snippet || '当前候选没有正文摘录。'}</p>
          </li>)}</ol> : <p className="chat-side-note">检索后将在这里展示可核对的 Issue 摘录。</p>}
        </section>
        <section className="block">
          <div className="block-label"><span className="cn">运行范围</span><span className="en">LOCAL SESSION</span></div>
          <p className="chat-side-note">本对话绑定 {activeRepository?.label || '未指定仓库'}，只查该仓库本地快照。会话保留最近 7 天；候选不能代替完整修复验证。</p>
          <p className="chat-side-note">{session ? `模型调用 ${session.model_calls} 次 · 检索 ${session.retrieval_calls} 次` : '等待会话建立'}</p>
          {session?.last_elapsed_ms != null && <p className="chat-side-note">上一轮耗时 {session.last_elapsed_ms} ms{session.prompt_tokens != null ? ` · 已记录输入 ${session.prompt_tokens} token` : ''}</p>}
          <button type="button" className="btn btn--sm" onClick={onOpenTriage}>转到单次分诊</button>
        </section>
          </div>
        </details>
      </div>
    </section>}
    </div>
  </main>
}
