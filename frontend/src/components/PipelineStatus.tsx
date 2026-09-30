import React from 'react'
import type { RunSnapshot } from '../types'

interface PipelineStatusProps {
  snapshot: RunSnapshot | null
  isAnalyzing: boolean
}

export const PipelineStatus: React.FC<PipelineStatusProps> = ({ snapshot, isAnalyzing }) => {
  if (!snapshot && !isAnalyzing) return null

  const nodes = snapshot?.nodes || [
    { name: 'query_analysis', label: '01 / QUERY ANALYSIS', status: 'waiting' },
    { name: 'retrieval', label: '02 / RETRIEVAL', status: 'waiting' },
    { name: 'rerank', label: '03 / RERANK', status: 'waiting' },
    { name: 'decision', label: '04 / DECISION', status: 'waiting' },
  ]

  const analysis = snapshot?.analysis
  const rounds = snapshot?.rounds || []

  return (
    <section className="block">
      <div className="block-label">
        <span className="cn">执行流程</span>
        <span className="en">PIPELINE & EXECUTION TRACE</span>
      </div>

      <div className="pipeline-grid">
        {nodes.map((node, idx) => {
          const isNodeRunning = node.status === 'running'
          const isNodeCompleted = node.status === 'completed'
          return (
            <div
              key={node.name}
              className={`step-box ${isNodeRunning ? 'running' : ''} ${isNodeCompleted ? 'completed' : ''}`}
            >
              <div className="step-num">
                <span>STAGE 0{idx + 1}</span>
                <span className={`step-status-tag ${node.status}`}>
                  {node.status.toUpperCase()}
                </span>
              </div>
              <div className="step-name">{node.label.split(' / ')[1] || node.name}</div>
              <div className="step-time">
                {node.elapsed_ms !== undefined && node.elapsed_ms !== null
                  ? `${(node.elapsed_ms / 1000).toFixed(2)}s`
                  : isNodeRunning
                  ? 'RUNNING...'
                  : '—'}
              </div>
              {node.summary && (
                <div className="step-summary">
                  {node.summary}
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Query Analysis 详细产物展示 */}
      {analysis && (
        <div className="analysis-card">
          <div className="panel-kicker">
            Query Analysis 结构化输出
          </div>
          <div className="analysis-body">
            <div><b>改写检索词 (Rewritten Query)：</b> <span className="mono-inline">{analysis.rewritten_query}</span></div>
            {analysis.keywords.length > 0 && (
              <div className="analysis-row">
                <b>抽取关键词：</b>
                {analysis.keywords.map((kw, i) => (
                  <span key={i} className="tag">{kw}</span>
                ))}
              </div>
            )}
            <div className="analysis-row analysis-row--muted">
              <b>组件标签 (Component)：</b> {analysis.component || 'None'}
              {analysis.component_filter_note && (
                <span className="analysis-note">
                  ({analysis.component_filter_note})
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 重试回环 Timeline */}
      {rounds.length > 1 && (
        <div className="rounds-card">
          <div className="panel-kicker panel-kicker--accent">
            低置信度回环诊断记录 (Retry Rounds: {rounds.length})
          </div>
          {rounds.map((r, i) => (
            <div key={i} className="round-row">
              <span className="round-label">ROUND 0{r.round}：</span>
              <span>Decision={r.decision} (conf={r.confidence.toFixed(2)}), 召回={r.retrieved_count}条, 精排={r.candidate_count}条</span>
              {r.top_score !== undefined && r.top_score !== null && (
                <span className="round-score">
                  TopScore={r.top_score.toFixed(3)}
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  )
}
