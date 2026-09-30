import React, { useState } from 'react'
import type { CandidateInfo, RunSnapshot } from '../types'

interface CandidateListProps {
  snapshot: RunSnapshot | null
}

export const CandidateList: React.FC<CandidateListProps> = ({ snapshot }) => {
  const [expandedIds, setExpandedIds] = useState<Record<string, boolean>>({})

  if (!snapshot) return null

  // 优先展示 rerank candidates，若无则展示 retrieval candidates
  const candidates: CandidateInfo[] =
    snapshot.rerank?.candidates || snapshot.retrieval?.candidates || []

  if (candidates.length === 0) return null

  const toggleExpand = (id: string) => {
    setExpandedIds((prev) => ({ ...prev, [id]: !prev[id] }))
  }

  return (
    <section className="block">
      <div className="block-label">
        <span className="cn">候选证据</span>
        <span className="en">TOP CANDIDATES & MATCHED EVIDENCE</span>
      </div>

      <div className="panel-kicker">
        SHOWING TOP {candidates.length} CANDIDATES FROM HYBRID RETRIEVAL & CROSS-ENCODER RERANK:
      </div>

      {candidates.map((cand, idx) => {
        const isExpanded = expandedIds[cand.id] || cand.is_related
        return (
          <div
            key={cand.id}
            className={`candidate-item ${cand.is_related ? 'is-related' : ''}`}
          >
            <div className="candidate-head">
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                  <span className="tag tag--rank">
                    RANK #{idx + 1}
                  </span>
                  <span className="candidate-id">
                    {cand.id}
                  </span>
                  {cand.is_related && (
                    <span className="tag tag--strong">
                      MATCHED RELATED
                    </span>
                  )}
                </div>
                <div className="candidate-title">{cand.title || '(无标题 / 仅正文证据)'}</div>
              </div>

              <div className="candidate-scores">
                {cand.score !== undefined && cand.score !== null && (
                  <span className="candidate-score-pill">RRF: {cand.score.toFixed(4)}</span>
                )}
                {cand.rerank_score !== undefined && cand.rerank_score !== null && (
                  <span className="candidate-score-pill candidate-score-pill--rerank">
                    RERANK: {cand.rerank_score.toFixed(4)}
                  </span>
                )}
              </div>
            </div>

            {cand.body_snippet && (
              <div className="candidate-snippet">
                {isExpanded ? cand.body_snippet : `${cand.body_snippet.slice(0, 140)}...`}
              </div>
            )}

            <div style={{ marginTop: '8px', display: 'flex', justifyContent: 'flex-end' }}>
              <button
                type="button"
                className="btn btn--sm btn--xs"
                onClick={() => toggleExpand(cand.id)}
              >
                {isExpanded ? 'COLLAPSE EVIDENCE ▴' : 'EXPAND EVIDENCE ▾'}
              </button>
            </div>
          </div>
        )
      })}
    </section>
  )
}
