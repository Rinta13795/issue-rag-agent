import React, { useEffect, useState, useRef } from 'react'
import {
  createRun,
  fetchEvaluationSummary,
  fetchExamples,
  fetchSystemInfo,
  listenRunEvents,
} from './api'
import type { EvaluationSummary, ExampleItem, RunSnapshot, SystemInfo } from './types'
import { Header } from './components/Header'
import { IssueInput } from './components/IssueInput'
import { PipelineStatus } from './components/PipelineStatus'
import { DecisionSummary } from './components/DecisionSummary'
import { CandidateList } from './components/CandidateList'
import { EvaluationNotes } from './components/EvaluationNotes'
import { Footer } from './components/Footer'
import { ChatWorkspace } from './components/ChatWorkspace'

export const App: React.FC = () => {
  const [view, setView] = useState<'chat' | 'workspace' | 'knowledge' | 'quality'>('chat')
  const [examples, setExamples] = useState<ExampleItem[]>([])
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null)
  const [evaluation, setEvaluation] = useState<EvaluationSummary | null>(null)
  const [snapshot, setSnapshot] = useState<RunSnapshot | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const stopListeningRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    fetchExamples().then(setExamples).catch(console.error)
    fetchSystemInfo().then(setSystemInfo).catch(console.error)
    fetchEvaluationSummary().then(setEvaluation).catch(console.error)

    return () => {
      if (stopListeningRef.current) {
        stopListeningRef.current()
      }
    }
  }, [])

  const handleAnalyze = async (text: string, sampleId?: string) => {
    if (stopListeningRef.current) {
      stopListeningRef.current()
      stopListeningRef.current = null
    }

    setIsAnalyzing(true)
    setErrorMsg(null)
    setSnapshot(null)

    try {
      const res = await createRun(text, sampleId)
      const runId = res.run_id

      stopListeningRef.current = listenRunEvents(
        runId,
        (snap) => {
          setSnapshot(snap)
          if (snap.status === 'failed' && snap.error) {
            setErrorMsg(snap.error)
          }
        },
        (eventType, _data) => {
          console.log('[SSE Event]', eventType)
        },
        () => {
          setIsAnalyzing(false)
        },
        (err) => {
          console.error('SSE Error:', err)
          setIsAnalyzing(false)
        }
      )
    } catch (e: any) {
      setErrorMsg(e.message || '启动分析失败')
      setIsAnalyzing(false)
    }
  }

  const handleCopyJson = () => {
    if (snapshot) {
      navigator.clipboard.writeText(JSON.stringify(snapshot, null, 2))
      alert('已复制运行 JSON 快照到剪贴板')
    }
  }

  return (
    <ChatWorkspace activeView={view} onNavigate={setView} onOpenTriage={() => setView('workspace')}>
      <div className="page-container">
      {view === 'workspace' && <>
      <Header systemInfo={systemInfo} />

      <main>
      <div className="intake-layout">
      <IssueInput
        examples={examples}
        onAnalyze={handleAnalyze}
        isAnalyzing={isAnalyzing}
      />

      <aside className="intake-guide" aria-label="分诊说明">
        <div className="block-label"><span className="cn">Agent 执行计划</span><span className="en">PLAN</span></div>
        <div className="agent-presence"><span className="tag">{isAnalyzing ? '执行中' : snapshot?.status === 'completed' ? '已完成' : '等待任务'}</span><p>Issue Triage Agent</p><span className="sidebar-meta">自主检索 / 证据核验 / 分诊决策</span></div>
        <ol className="workflow-list">
          <li><span>01</span><div><b>理解问题</b><p>提取异常特征与组件信息。</p></div></li>
          <li><span>02</span><div><b>检索历史记录</b><p>查找描述相近、现象相关的问题。</p></div></li>
          <li><span>03</span><div><b>核验候选证据</b><p>重排相关记录，补齐问题原文。</p></div></li>
          <li><span>04</span><div><b>形成分诊建议</b><p>结合候选证据，辅助判断是否重复。</p></div></li>
        </ol>
        <p className="guide-note">建议由研发负责人结合原文证据确认后，再决定关联或关闭 Issue。</p>
      </aside>
      </div>

      {!snapshot && !isAnalyzing && !errorMsg && (
        <section className="block result-placeholder">
          <div className="block-label"><span className="cn">分诊结果</span><span className="en">TRIAGE REVIEW</span></div>
          <div className="placeholder-content"><b>等待提交问题</b><p>分析完成后，这里将展示分诊结论、判断依据与关联问题，帮助你决定下一步处理方式。</p></div>
        </section>
      )}

      {errorMsg && (
        <div className="app-alert">
          <div className="app-alert-title">
            ■ 运行异常提示
          </div>
          <div className="app-alert-body">{errorMsg}</div>
        </div>
      )}

      <PipelineStatus snapshot={snapshot} isAnalyzing={isAnalyzing} />

      <DecisionSummary snapshot={snapshot} />

      <CandidateList snapshot={snapshot} />

      {snapshot && snapshot.status === 'completed' && (
        <div className="app-actions">
          <button type="button" className="btn btn--sm" onClick={handleCopyJson}>
            复制分析记录（JSON）
          </button>
        </div>
      )}

      <EvaluationNotes evaluation={evaluation} />

      </main>
      </>}
      {view === 'knowledge' && <section className="block">
        <div className="block-label"><span className="cn">知识与数据源</span><span className="en">KNOWLEDGE SOURCES</span></div>
        <h1 className="masthead-title">Agent 的知识底座<span className="dot">。</span></h1>
        <p className="masthead-sub">连接历史问题，保留可追溯的原文证据。</p>
        <dl className="source-list">
          <div><dt>问题语料</dt><dd>{systemInfo ? `${systemInfo.dataset_name} / ${systemInfo.dataset_size.toLocaleString()} 条` : '等待后端连接'}</dd></div>
          <div><dt>关键词索引</dt><dd>{systemInfo ? systemInfo.bm25_ready ? '已就绪' : '未就绪' : '状态未知'}</dd></div>
          <div><dt>语义索引</dt><dd>{systemInfo ? systemInfo.vector_ready ? '已就绪' : '未就绪' : '状态未知'}</dd></div>
          <div><dt>原文证据库</dt><dd>{systemInfo ? systemInfo.docstore_ready ? '已就绪' : '未就绪' : '状态未知'}</dd></div>
        </dl>
      </section>}
      {view === 'quality' && <section className="block">
        <div className="block-label"><span className="cn">质量评估</span><span className="en">AGENT EVALUATION</span></div>
        <h1 className="masthead-title">每一次判断，有据可查。</h1>
        <p className="masthead-sub">对照检索基线，检查召回质量与排序效果。</p>
        <div className="quality-content">{evaluation ? <EvaluationNotes evaluation={evaluation} /> : <p className="guide-note">尚未加载评测报告。连接后端后可查看召回率、排序指标和评测边界。</p>}</div>
      </section>}
      {view !== 'chat' && <Footer />}
      </div>
    </ChatWorkspace>
  )
}

export default App
