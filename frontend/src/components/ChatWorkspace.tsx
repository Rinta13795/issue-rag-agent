import { useCallback, useEffect, useRef, useState } from 'react'
import { ChatApiError, confirmMemory, createChatSession, createIssueDraft, deleteMemory, editIssueDraft, getChatSession, getRepositorySync, listChatRepositories, listChatSessions, listMemories, previewChatIssue, proposeMemory, publishIssueDraft, retryChatMessage, sendChatMessage, syncChatRepository } from '../api'
import type { ChatRepository, ChatSession, ChatSessionSummary, MemoryRecord } from '../types'

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
  const [sessions, setSessions] = useState<ChatSessionSummary[]>([])
  const [repositories, setRepositories] = useState<ChatRepository[]>([])
  const [selectedRepository, setSelectedRepository] = useState('')
  const [githubRepository, setGithubRepository] = useState('')
  const [issueLink, setIssueLink] = useState('')
  const [syncStatus, setSyncStatus] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [issueTitle, setIssueTitle] = useState('')
  const [issueBody, setIssueBody] = useState('')
  const [publishConfirmed, setPublishConfirmed] = useState(false)
  const [memoryText, setMemoryText] = useState('')
  const [memoryKind, setMemoryKind] = useState<'preference' | 'experience'>('experience')
  const [memoryScope, setMemoryScope] = useState<'global' | 'repository'>('repository')
  const [memories, setMemories] = useState<MemoryRecord[]>([])
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

  useEffect(() => {
    setIssueTitle(session?.issue_draft?.title || '')
    setIssueBody(session?.issue_draft?.body || '')
    setPublishConfirmed(false)
  }, [session?.issue_draft?.draft_id, session?.issue_draft?.version, session?.issue_draft?.title, session?.issue_draft?.body])

  useEffect(() => {
    setMemoryText(session?.memory_proposal?.text || '')
    setMemoryKind(session?.memory_proposal?.kind || 'experience')
    setMemoryScope(session?.memory_proposal?.scope || 'repository')
  }, [session?.memory_proposal?.text, session?.memory_proposal?.kind, session?.memory_proposal?.scope])

  useEffect(() => {
    if (!session?.repository_id) { setMemories([]); return }
    void listMemories(session.repository_id).then(setMemories).catch(() => setMemories([]))
  }, [session?.repository_id])

  const startNew = () => {
    selectionEpochRef.current += 1
    window.localStorage.removeItem(STORAGE_KEY)
    setSession(null)
    setSelectedRepository('')
    setGithubRepository('')
    setIssueLink('')
    setSyncStatus(null)
    setDraft('')
    setIssueTitle('')
    setIssueBody('')
    setPublishConfirmed(false)
    setMemoryText('')
    setMemories([])
    setError(null)
    setNotice(null)
  }

  const createForRepository = async (repositoryId = selectedRepository, issueUrl?: string) => {
    if (!repositoryId) {
      setError('请先选择一个已有仓库快照。')
      const picker = document.getElementById('chat-repository')
      picker?.scrollIntoView({ behavior: 'smooth', block: 'center' })
      picker?.focus()
      return
    }
    const createEpoch = ++selectionEpochRef.current
    try {
      const created = await createChatSession(repositoryId, issueUrl)
      if (selectionEpochRef.current !== createEpoch) return
      window.localStorage.setItem(STORAGE_KEY, created.session_id)
      setSession(created)
      setError(null)
      setNotice(null)
      await refreshSessions()
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '暂时无法创建会话')
    }
  }

  const importIssue = async () => {
    if (!issueLink.trim()) {
      setError('请先粘贴 GitHub Issue 链接，再读取。')
      document.getElementById('chat-issue-link')?.focus()
      return
    }
    const importEpoch = ++selectionEpochRef.current
    setError(null)
    setSyncStatus('正在读取源 Issue…')
    try {
      const source = await previewChatIssue(issueLink.trim())
      if (selectionEpochRef.current !== importEpoch) return
      let job = await syncChatRepository(source.repository)
      while (selectionEpochRef.current === importEpoch && job.status !== 'completed' && job.status !== 'failed') {
        setSyncStatus(job.message)
        await new Promise((resolve) => window.setTimeout(resolve, 1200))
        if (selectionEpochRef.current !== importEpoch) return
        job = await getRepositorySync(job.job_id)
      }
      if (selectionEpochRef.current !== importEpoch) return
      if (job.status === 'failed' || !job.repository_id) throw new Error(job.message)
      setRepositories(await listChatRepositories())
      if (selectionEpochRef.current !== importEpoch) return
      setSyncStatus(null)
      await createForRepository(job.repository_id, source.url)
    } catch (caught) {
      if (selectionEpochRef.current !== importEpoch) return
      setSyncStatus(null)
      setError(caught instanceof Error ? caught.message : 'Issue 导入失败')
    }
  }

  const syncAndCreate = async () => {
    if (!githubRepository.trim()) {
      setError('请先输入公开仓库名，格式为 owner/repo。')
      document.getElementById('chat-github-repository')?.focus()
      return
    }
    const syncEpoch = ++selectionEpochRef.current
    setError(null)
    setSyncStatus('正在提交仓库…')
    try {
      let job = await syncChatRepository(githubRepository.trim())
      while (selectionEpochRef.current === syncEpoch && job.status !== 'completed' && job.status !== 'failed') {
        setSyncStatus(job.message)
        await new Promise((resolve) => window.setTimeout(resolve, 1200))
        if (selectionEpochRef.current !== syncEpoch) return
        job = await getRepositorySync(job.job_id)
      }
      if (selectionEpochRef.current !== syncEpoch) return
      if (job.status === 'failed' || !job.repository_id) throw new Error(job.message)
      setRepositories(await listChatRepositories())
      if (selectionEpochRef.current !== syncEpoch) return
      setSyncStatus(null)
      await createForRepository(job.repository_id)
    } catch (caught) {
      if (selectionEpochRef.current !== syncEpoch) return
      setSyncStatus(null)
      setError(caught instanceof Error ? caught.message : '仓库同步失败')
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

  const retry = async () => {
    if (!session || session.status !== 'failed' || isSending) return
    setIsSending(true)
    setError(null)
    try {
      await retryChatMessage(session.session_id)
      const updated = await getChatSession(session.session_id)
      if (window.localStorage.getItem(STORAGE_KEY) === session.session_id) setSession(updated)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '重试失败')
    } finally {
      setIsSending(false)
    }
  }

  const makeIssueDraft = async () => {
    if (!session || isSending) return
    setIsSending(true)
    setError(null)
    try {
      setSession(await createIssueDraft(session.session_id))
      setNotice('草稿已生成，尚未发布。请核对内容与可能重复的反馈。')
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '草稿生成失败')
    } finally { setIsSending(false) }
  }

  const saveIssueDraft = async () => {
    if (!session?.issue_draft || isSending) return
    setIsSending(true)
    setError(null)
    try {
      setSession(await editIssueDraft(session.session_id, issueTitle.trim(), issueBody.trim(), session.issue_draft.version))
      setNotice('草稿已保存，仍未发布。')
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '保存草稿失败')
    } finally { setIsSending(false) }
  }

  const publishIssue = async () => {
    if (!session?.issue_draft || !publishConfirmed || isSending) return
    setIsSending(true)
    setError(null)
    try {
      setSession(await publishIssueDraft(session.session_id, session.issue_draft.draft_id, session.issue_draft.version))
      setNotice('Issue 已发布到 GitHub。')
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '发布失败；草稿仍在')
      const latest = await getChatSession(session.session_id).catch(() => null)
      if (latest) setSession(latest)
    } finally { setIsSending(false) }
  }

  const makeMemoryProposal = async () => {
    if (!session || isSending) return
    setIsSending(true)
    setError(null)
    try {
      setSession(await proposeMemory(session.session_id))
      setNotice('这是一条待确认的记忆，尚未长期保存。')
    } catch (caught) { setError(caught instanceof Error ? caught.message : '提炼失败') }
    finally { setIsSending(false) }
  }

  const saveMemory = async () => {
    if (!session?.memory_proposal || !memoryText.trim() || isSending) return
    setIsSending(true)
    setError(null)
    try {
      await confirmMemory(session.session_id, memoryText.trim(), memoryKind, memoryScope)
      setSession(await getChatSession(session.session_id))
      setMemories(await listMemories(session.repository_id || undefined))
      setNotice('这条经验已保存；后续相关对话会按需使用。')
    } catch (caught) { setError(caught instanceof Error ? caught.message : '保存记忆失败') }
    finally { setIsSending(false) }
  }

  const removeMemory = async (memoryId: string) => {
    setError(null)
    try {
      await deleteMemory(memoryId)
      setMemories((existing) => existing.filter((item) => item.memory_id !== memoryId))
    } catch (caught) { setError(caught instanceof Error ? caught.message : '删除记忆失败') }
  }

  const busy = isSending || Boolean(session && activeStates.has(session.status))
  const activeRepository = repositories.find((repo) => repo.id === session?.repository_id)
  const issueUrl = (id: string): string | null => {
    const issueNumber = id.startsWith(`${session?.repository_id}:`) ? id.slice((session?.repository_id || '').length + 1) : ''
    return activeRepository?.github_url && /^\d+$/.test(issueNumber) ? `${activeRepository.github_url}/issues/${issueNumber}` : null
  }
  const canSendDraft = Boolean(draft.trim() && draft.trim() !== '报错原文：')
  const issueDraftDirty = Boolean(session?.issue_draft && (issueTitle !== session.issue_draft.title || issueBody !== session.issue_draft.body))
  const hasConversation = Boolean(session?.messages.length || session?.source_issue) || busy
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
        <span>{item.title}</span><small>{repositories.find((repo) => repo.id === item.repository_id)?.label || item.repository_id || '旧会话 · 未绑定仓库'}</small>
      </button>)}
      {!sessions.length && <p className="chat-side-note">还没有对话。</p>}
    </aside>
    <div className="chat-conversation-area">
    {!hasConversation && <section className="chat-home" aria-label="开始对话">
      <div className="masthead-pre">ISSUE TRIAGE / CONVERSATION</div>
      <h1 className="masthead-title">从一个 Issue 开始<span className="dot">。</span></h1>
      <p className="masthead-sub">贴上已有 Issue 链接，Agent 会读取原文并围绕它与你排查；尚未提交的问题，也可以先选仓库再描述。</p>
      {!session && <div className="chat-repository-picker">
        <div className="chat-repository-add">
          <label className="field-label" htmlFor="chat-issue-link">导入已有 GitHub Issue</label>
          <input id="chat-issue-link" className="field-input" value={issueLink} onChange={(event) => setIssueLink(event.target.value)} placeholder="https://github.com/owner/repo/issues/123" />
          <button type="button" className="btn btn--primary" disabled={Boolean(syncStatus)} onClick={() => void importIssue()}>读取 Issue 并开始对话</button>
        </div>
        <div className="chat-repository-add">
          <label className="field-label" htmlFor="chat-github-repository">还没有 Issue？先选择仓库</label>
          <input id="chat-github-repository" className="field-input" value={githubRepository} onChange={(event) => setGithubRepository(event.target.value)} placeholder="owner/repo，例如 HKUDS/OpenHarness" />
          <p className="chat-side-note">首次会同步最近更新的最多 300 条公开 Issue，建立本地快照；已同步仓库会直接复用。不会实时读取 PR、代码或文档。</p>
          <button type="button" className="btn" disabled={Boolean(syncStatus)} onClick={() => void syncAndCreate()}>同步并开始对话</button>
          {syncStatus && <p className="chat-connection" role="status">{syncStatus}。首次索引可能需要几分钟。</p>}
        </div>
        <div className="chat-repository-existing">
          <label className="field-label" htmlFor="chat-repository">或使用已有快照</label>
          <select id="chat-repository" value={selectedRepository} onChange={(event) => setSelectedRepository(event.target.value)}>
            <option value="">请选择已有快照</option>
            {repositories.map((repo) => <option key={repo.id} value={repo.id}>{repo.label} · {repo.issue_count.toLocaleString()} 条 Issue</option>)}
          </select>
          {selectedRepository && <p className="chat-side-note">{repositories.find((repo) => repo.id === selectedRepository)?.source}；仓库选定后，本对话中不可切换。</p>}
          <button type="button" className="btn btn--primary" disabled={Boolean(syncStatus)} onClick={() => void createForRepository()}>开始对话</button>
        </div>
      </div>}
      {notice && <p className="chat-connection" role="status">{notice}</p>}
      {error && <div className="chat-connection-error" role="alert"><p className="chat-error">■ {error}</p><button type="button" className="btn btn--sm" onClick={startNew}>重新选择</button></div>}
      {session && <p className="chat-connection">当前仓库：{activeRepository?.label || '旧会话'} · 本地快照与 GitHub 实时搜索都限定在此仓库</p>}
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
        {session?.source_issue && <section className="chat-source-issue" aria-label="当前源 Issue">
          <div className="block-label"><span className="cn">当前 Issue</span><span className="en">SOURCE ISSUE · #{session.source_issue.number}</span></div>
          <a className="chat-source-title" href={session.source_issue.url} target="_blank" rel="noreferrer">{session.source_issue.title} ↗</a>
          <p className="chat-side-note">{session.source_issue.repository} · {session.source_issue.state} · 已读取 {session.source_issue.comments.length} 条评论{session.source_issue.comments_truncated ? '，还有未载入评论' : ''}</p>
          <p className="chat-source-body">{session.source_issue.body || '这条 Issue 没有正文。'}</p>
          {session.source_issue.comments.length > 0 && <details className="chat-source-comments"><summary>查看已读取的讨论</summary>
            {session.source_issue.comments.map((comment, index) => <p key={`${comment.url}-${index}`}><a href={comment.url} target="_blank" rel="noreferrer">{comment.author} ↗</a>：{comment.body}</p>)}
          </details>}
        </section>}
        <div className="chat-messages" aria-live="polite">
          {session?.messages.map((message, index) => <article className={`chat-message chat-message--${message.role}`} key={message.id}>
            <div className="chat-message-meta">{message.role === 'user' ? '你' : 'Issue Agent'} · {message.action === 'retrieve' ? '已检索证据' : message.action === 'clarify' ? '需要澄清' : message.action === 'reply' ? '基于当前上下文' : '对话'}</div>
            <p>{message.content}</p>
            {message.citations.length > 0 && <div className="chat-citation">依据：{message.citations.map((id, index) => <span key={id}>{index > 0 ? ' · ' : ''}{issueUrl(id) ? <a href={issueUrl(id)!} target="_blank" rel="noreferrer">{id} ↗</a> : id}</span>)}</div>}
            {message.action === 'clarify' && index === session.messages.length - 1 && !busy && <div className="chat-clarify-options" aria-label="下一步选择">
              {session.retrieval_calls === 0 && canSearchHistory && <button type="button" className="btn btn--sm" onClick={() => void send('请先用我已经描述的故障现象搜索本地历史 Issue。')}>先查历史 Issue</button>}
              {session.retrieval_calls > 0 ? <>
                <button type="button" className="btn btn--sm" onClick={() => prefill('报错原文：\n')}>补充报错原文</button>
                <button type="button" className="btn btn--sm" onClick={() => setNotice('先查本地快照，再尝试 GitHub 实时搜索；结果仅属于当前仓库，未命中不等于没有解法。')}>查看检索范围</button>
              </> : <button type="button" className="btn btn--sm" onClick={() => prefill(draft)}>补充线索</button>}
            </div>}
          </article>)}
          {busy && <div className="chat-progress" role="status">{statusLabels[session?.status || ''] || '正在发送'}…</div>}
          <div ref={endRef} />
        </div>
        {(error || session?.last_error) && <div className="chat-connection-error" role="alert">
          <p className="chat-error">■ {error || session?.last_error}</p>
          {session?.status === 'failed' && <button type="button" className="btn btn--sm" disabled={isSending} onClick={() => void retry()}>重试这一轮</button>}
        </div>}
        {notice && <p className="chat-connection" role="status">{notice}</p>}
        {session?.repository_id ? composer : <p className="chat-side-note">这是升级前未绑定仓库的旧对话。请点“新对话”并选择仓库后继续。</p>}
        {session?.repository_id && <section className="chat-issue-action block">
          <div className="block-label"><span className="cn">处理 Issue</span><span className="en">DRAFT · REVIEW · PUBLISH</span></div>
          {!session.issue_draft && <button type="button" className="btn" disabled={busy} onClick={() => void makeIssueDraft()}>起草新 Issue</button>}
          {session.issue_draft && <>
            <p className="chat-side-note">目标仓库：{activeRepository?.label || '未知仓库'}。发布后内容将公开显示。当前状态：{session.issue_draft.status === 'published' ? '已发布' : session.issue_draft.status === 'uncertain' ? '结果待核验' : '未发布'}。</p>
            {session.source_issue && <p className="chat-side-note">当前已经有源 Issue #{session.source_issue.number}。如果只是补充线索，请优先在原 Issue 讨论；这里发布会创建另一条新 Issue。</p>}
            {session.issue_draft.search_status !== 'ok' && <p className="chat-side-note">GitHub 实时查重{session.issue_draft.search_status === 'failed' ? '失败' : '未运行'}；不能确认仓库是否已有相同反馈。</p>}
            {session.issue_draft.possible_duplicates.length > 0 && <div className="chat-draft-duplicates"><p className="chat-side-note">发布前请核对可能相关的现有 Issue：</p>
              {session.issue_draft.possible_duplicates.map((candidate) => <a key={candidate.id} href={candidate.url || issueUrl(candidate.id) || '#'} target="_blank" rel="noreferrer">{candidate.id} · {candidate.title} ↗</a>)}
            </div>}
            <label className="field-label" htmlFor="chat-issue-title">标题</label>
            <input id="chat-issue-title" className="field-input" value={issueTitle} maxLength={256} disabled={session.issue_draft.status !== 'draft'} onChange={(event) => { setIssueTitle(event.target.value); setPublishConfirmed(false) }} />
            <label className="field-label" htmlFor="chat-issue-body">正文</label>
            <textarea id="chat-issue-body" className="field-textarea" value={issueBody} maxLength={20000} disabled={session.issue_draft.status !== 'draft'} onChange={(event) => { setIssueBody(event.target.value); setPublishConfirmed(false) }} />
            {session.issue_draft.status === 'draft' && <div className="chat-draft-actions">
              <button type="button" className="btn" disabled={busy || !issueTitle.trim() || !issueBody.trim() || !issueDraftDirty} onClick={() => void saveIssueDraft()}>保存草稿</button>
              <label className="chat-publish-check"><input type="checkbox" checked={publishConfirmed} onChange={(event) => setPublishConfirmed(event.target.checked)} />我已核对仓库和完整内容，同意公开发布</label>
              <button type="button" className="btn btn--primary" disabled={busy || issueDraftDirty || !publishConfirmed || !activeRepository?.github_url} onClick={() => void publishIssue()}>确认发布到 GitHub</button>
            </div>}
            {session.issue_draft.status === 'published' && session.issue_draft.published_url && <a className="chat-source-title" href={session.issue_draft.published_url} target="_blank" rel="noreferrer">查看已发布的 Issue #{session.issue_draft.published_number} ↗</a>}
            {session.issue_draft.status === 'uncertain' && <p className="chat-side-note">网络中断后无法确认是否创建成功。请先打开目标仓库核验，勿重复发布。</p>}
          </>}
        </section>}
        {session?.repository_id && <section className="chat-memory block">
          <div className="block-label"><span className="cn">处理经验</span><span className="en">CONFIRMED MEMORY</span></div>
          <p className="chat-side-note">只有你核对并保存的偏好或结果才跨对话复用；聊天记录不会自动变成长期记忆。</p>
          <button type="button" className="btn" disabled={busy || (!session.source_issue && !session.messages.length)} onClick={() => void makeMemoryProposal()}>提炼本次经验</button>
          {session.memory_proposal && <div className="chat-memory-proposal">
            <p className="chat-side-note">依据原文：{session.memory_proposal.source_excerpt}</p>
            <label className="field-label" htmlFor="chat-memory-text">待确认的记忆</label>
            <textarea id="chat-memory-text" className="field-textarea" value={memoryText} maxLength={500} onChange={(event) => setMemoryText(event.target.value)} />
            <label className="field-label" htmlFor="chat-memory-kind">类型</label>
            <select id="chat-memory-kind" value={memoryKind} onChange={(event) => { const kind = event.target.value as 'preference' | 'experience'; setMemoryKind(kind); if (kind === 'experience') setMemoryScope('repository') }}>
              <option value="experience">已确认的仓库处理经验</option><option value="preference">我的处理偏好</option>
            </select>
            {memoryKind === 'preference' && <label className="chat-publish-check"><input type="checkbox" checked={memoryScope === 'global'} onChange={(event) => setMemoryScope(event.target.checked ? 'global' : 'repository')} />对其他仓库也适用</label>}
            <button type="button" className="btn btn--primary" disabled={busy || !memoryText.trim()} onClick={() => void saveMemory()}>确认保存这条经验</button>
          </div>}
          {memories.length > 0 && <div className="chat-memory-list">{memories.map((item) => <div key={item.memory_id}>
            <p>{item.text}</p><small>{item.kind === 'preference' ? '处理偏好' : '仓库经验'} · {item.repository_id ? '当前仓库' : '跨仓库'}</small>
            <button type="button" className="btn btn--sm" onClick={() => void removeMemory(item.memory_id)}>忘记</button>
          </div>)}</div>}
        </section>}
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
            <span className="chat-candidate-id">{candidate.url || issueUrl(candidate.id) ? <a href={(candidate.url || issueUrl(candidate.id))!} target="_blank" rel="noreferrer">{candidate.id} ↗ 查看原文</a> : candidate.id} · {candidate.source === 'github' ? 'GitHub 实时' : '本地快照'}</span>
            <b>{candidate.title || '无标题 Issue'}</b>
            <p>{candidate.body_snippet || '当前候选没有正文摘录。'}</p>
          </li>)}</ol> : <p className="chat-side-note">检索后将在这里展示可核对的 Issue 摘录。</p>}
        </section>
        <section className="block">
          <div className="block-label"><span className="cn">运行范围</span><span className="en">LOCAL SESSION</span></div>
          <p className="chat-side-note">本对话绑定 {activeRepository?.label || '未指定仓库'}。先查本地快照；GitHub 仓库还会尝试实时搜索。会话保留最近 7 天，候选不能代替完整修复验证。</p>
          {session?.live_search_status === 'failed' && <p className="chat-side-note">实时搜索未完成：{session.live_search_message || 'GitHub 暂时不可用'}。不能据此认定仓库没有相关 Issue。</p>}
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
