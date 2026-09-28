import { useEffect, useRef, useState } from 'react'
import { createChatSession, getChatSession, sendChatMessage } from '../api'
import type { ChatSession } from '../types'

const STORAGE_KEY = 'issue-rag-local-chat-session'
const activeStates = new Set(['thinking', 'retrieving', 'answering'])
const statusLabels: Record<string, string> = {
  thinking: '正在理解这轮对话',
  retrieving: '正在查找历史 Issue',
  answering: '正在整理证据与回答',
}

interface Props {
  onOpenTriage: () => void
}

export function ChatWorkspace({ onOpenTriage }: Props) {
  const [session, setSession] = useState<ChatSession | null>(null)
  const [draft, setDraft] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isSending, setIsSending] = useState(false)
  const loadedRef = useRef(false)
  const endRef = useRef<HTMLDivElement | null>(null)
  const sessionId = session?.session_id
  const sessionStatus = session?.status

  useEffect(() => {
    if (loadedRef.current) return
    loadedRef.current = true
    const savedId = window.localStorage.getItem(STORAGE_KEY)
    const load = async () => {
      try {
        const current = savedId ? await getChatSession(savedId) : await createChatSession()
        window.localStorage.setItem(STORAGE_KEY, current.session_id)
        setSession(current)
      } catch {
        if (savedId) {
          window.localStorage.removeItem(STORAGE_KEY)
          setError('之前的本地会话已失效，可以开始一段新对话。')
        } else {
          setError('暂时无法连接对话服务，请确认后端已经启动。')
        }
      }
    }
    void load()
  }, [])

  useEffect(() => {
    if (!sessionId || !sessionStatus || !activeStates.has(sessionStatus)) return
    const timer = window.setInterval(() => {
      getChatSession(sessionId).then(setSession).catch(() => {
        setError('会话连接中断，请开始新对话。')
      })
    }, 900)
    return () => window.clearInterval(timer)
  }, [sessionId, sessionStatus])

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [session?.messages.length])

  const startNew = async () => {
    try {
      const created = await createChatSession()
      window.localStorage.setItem(STORAGE_KEY, created.session_id)
      setSession(created)
      setDraft('')
      setError(null)
    } catch {
      setError('暂时无法创建会话，请确认后端已经启动。')
    }
  }

  const send = async () => {
    if (!session || activeStates.has(session.status) || isSending || !draft.trim()) return
    const content = draft.trim()
    setIsSending(true)
    setError(null)
    try {
      await sendChatMessage(session.session_id, content, window.crypto.randomUUID())
      setDraft('')
      setSession(await getChatSession(session.session_id))
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '消息发送失败')
    } finally {
      setIsSending(false)
    }
  }

  const busy = isSending || Boolean(session && activeStates.has(session.status))

  return <main className="chat-workspace">
    <div className="masthead">
      <div className="masthead-pre">ISSUE TRIAGE / CONVERSATION</div>
      <h1 className="masthead-title">把问题讲给 Agent 听<span className="dot">。</span></h1>
      <p className="masthead-sub">像和同事讨论一样补充细节、追问候选；需要历史证据时，Agent 再去检索。</p>
      <div className="masthead-rule" />
    </div>

    <div className="chat-layout">
      <section className="chat-main" aria-label="对话">
        <div className="chat-section-head">
          <div className="block-label"><span className="cn">当前对话</span><span className="en">CONVERSATION</span></div>
          <button type="button" className="btn btn--sm" onClick={startNew}>新对话</button>
        </div>
        {!session && !error && <p className="chat-empty">正在连接本地会话…</p>}
        {session && session.messages.length === 0 && <div className="chat-empty">
          <b>先说说你遇到了什么。</b>
          <p>一句话、报错、日志都可以。信息不足时我会追问；补充新线索后再查历史 Issue。</p>
        </div>}
        <div className="chat-messages" aria-live="polite">
          {session?.messages.map((message) => <article className={`chat-message chat-message--${message.role}`} key={message.id}>
            <div className="chat-message-meta">{message.role === 'user' ? '你' : 'Issue Agent'} · {message.action === 'retrieve' ? '已检索证据' : message.action === 'clarify' ? '需要澄清' : message.action === 'reply' ? '基于当前上下文' : '对话'}</div>
            <p>{message.content}</p>
            {message.citations.length > 0 && <div className="chat-citation">依据：{message.citations.join(' · ')}</div>}
          </article>)}
          {busy && <div className="chat-progress" role="status">{statusLabels[session?.status || ''] || '正在发送'}…</div>}
          <div ref={endRef} />
        </div>
        {(error || session?.last_error) && <p className="chat-error">■ {error || session?.last_error}</p>}
        <div className="chat-composer">
          <label className="field-label" htmlFor="chat-input">继续说</label>
          <textarea
            id="chat-input" className="field-textarea" value={draft} maxLength={12000}
            placeholder="描述现象，或追问：第一条为什么像？"
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void send() } }}
          />
          <div className="chat-composer-foot"><span>Enter 发送 · Shift + Enter 换行</span><button type="button" className="btn btn--primary" disabled={!session || busy || !draft.trim()} onClick={() => void send()}>发送消息</button></div>
        </div>
      </section>

      <aside className="chat-side" aria-label="问题卡片与候选证据">
        <section className="block">
          <div className="block-label"><span className="cn">已知线索</span><span className="en">USER FACTS</span></div>
          {session?.facts.length ? <ul className="chat-facts">{session.facts.map((fact, index) => <li key={`${fact.source_message_id}-${index}`}>{fact.value}</li>)}</ul> : <p className="chat-side-note">只记录你明确说出的故障线索，不把模型猜测写入问题卡片。</p>}
          {session?.open_question && <p className="chat-open-question">还需确认：{session.open_question}</p>}
        </section>
        <section className="block">
          <div className="block-label"><span className="cn">历史候选</span><span className="en">ISSUE EVIDENCE</span></div>
          {session?.candidates.length ? <ol className="chat-candidates">{session.candidates.map((candidate) => <li key={candidate.id}>
            <span className="chat-candidate-id">{candidate.id}</span>
            <b>{candidate.title || '无标题 Issue'}</b>
            <p>{candidate.body_snippet || '当前候选没有正文摘录。'}</p>
          </li>)}</ol> : <p className="chat-side-note">检索后将在这里展示可核对的 Issue 摘录。</p>}
        </section>
        <section className="block">
          <div className="block-label"><span className="cn">运行范围</span><span className="en">LOCAL SESSION</span></div>
          <p className="chat-side-note">本地会话，服务重启后会失效；候选来自历史 Issue，不能代替完整修复验证。</p>
          <p className="chat-side-note">{session ? `模型调用 ${session.model_calls} 次 · 检索 ${session.retrieval_calls} 次` : '等待会话建立'}</p>
          {session?.last_elapsed_ms != null && <p className="chat-side-note">上一轮耗时 {session.last_elapsed_ms} ms{session.prompt_tokens != null ? ` · 已记录输入 ${session.prompt_tokens} token` : ''}</p>}
          <button type="button" className="btn btn--sm" onClick={onOpenTriage}>转到单次分诊</button>
        </section>
      </aside>
    </div>
  </main>
}
