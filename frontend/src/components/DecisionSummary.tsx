import React from 'react'
import type { RunSnapshot } from '../types'

interface DecisionSummaryProps {
  snapshot: RunSnapshot | null
}

export const DecisionSummary: React.FC<DecisionSummaryProps> = ({ snapshot }) => {
  if (!snapshot || !snapshot.decision) return null

  const dec = snapshot.decision
  const totalSeconds = snapshot.total_elapsed_ms ? (snapshot.total_elapsed_ms / 1000).toFixed(2) : null

  return (
    <section className="block">
      <div className="block-label">
        <span className="cn">分诊判断</span>
        <span className="en">DECISION & REASONING</span>
      </div>

      <div className="decision-panel">
        <div className="decision-header">
          <div className="decision-badge-group">
            <span className="field-label" style={{ marginBottom: 0 }}>TRIAGE VERDICT:</span>
            <span className={`decision-val ${dec.decision.toLowerCase()}`}>
              {dec.decision.toUpperCase()}
            </span>
          </div>

          <div className="decision-metrics">
            <div className="decision-metric-item">
              CONFIDENCE: <b>{dec.confidence.toFixed(2)}</b>
            </div>
            {totalSeconds && (
              <div className="decision-metric-item">
                TOTAL LATENCY: <b>{totalSeconds}s</b>
              </div>
            )}
            <div className="decision-metric-item">
              ROUNDS: <b>{dec.retry_count}</b>
            </div>
          </div>
        </div>

        {dec.related_issues.length > 0 && (
          <div style={{ marginTop: '16px', display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <span className="field-label" style={{ marginBottom: 0 }}>关联命中 ID：</span>
            {dec.related_issues.map((id) => (
              <span key={id} className="tag" style={{ borderColor: 'var(--ink)', color: 'var(--ink)', fontWeight: 600 }}>
                #{id}
              </span>
            ))}
          </div>
        )}

        <div className="decision-reasoning">
          <p><b>判定依据 (Reasoning)：</b> {dec.reasoning}</p>
        </div>
      </div>
    </section>
  )
}
