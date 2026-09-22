import React, { useState } from 'react'
import type { EvaluationSummary } from '../types'

interface EvaluationNotesProps {
  evaluation: EvaluationSummary | null
}

export const EvaluationNotes: React.FC<EvaluationNotesProps> = ({ evaluation }) => {
  const [isOpen, setIsOpen] = useState(false)

  if (!evaluation) return null

  const rows = [
    { key: 'vector_only', label: 'Vector Only (MiniLM-L6-v2)' },
    { key: 'bm25_only', label: 'BM25 Only (Issue-level)' },
    { key: 'hybrid', label: 'Hybrid RRF (Dual-Query)' },
    { key: 'hybrid_rerank', label: 'Hybrid + Rerank (bge-base)' },
  ]

  return (
    <section className="block">
      <div
        className="block-label"
        style={{ cursor: 'pointer', display: 'flex', justifyContent: 'space-between' }}
        onClick={() => setIsOpen(!isOpen)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span className="cn">评测基准与边界</span>
          <span className="en">BENCHMARK (N={evaluation.sample_size}) & SYSTEM LIMITS</span>
        </div>
        <button type="button" className="btn btn--sm" style={{ fontSize: '9px', padding: '2px 8px' }}>
          {isOpen ? 'HIDE BENCHMARK ▴' : 'VIEW BENCHMARK ▾'}
        </button>
      </div>

      {isOpen && (
        <div style={{ border: '1px solid var(--rule)', borderRadius: '5px', padding: '20px', background: 'var(--paper)' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: '11px', color: 'var(--ink-faint)', marginBottom: '10px' }}>
            HELD-OUT TEST SET (200 QUERIES, SEED=42, SELF-HIT EXCLUDED):
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>RETRIEVAL CONFIG</th>
                  <th>RECALL@5</th>
                  <th>RECALL@10</th>
                  <th>MRR</th>
                  <th>NDCG@10</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const m = evaluation.metrics[r.key] || {}
                  return (
                    <tr key={r.key}>
                      <td style={{ fontWeight: 500 }}>{r.label}</td>
                      <td>{m['recall@5'] ? m['recall@5'].toFixed(4) : '—'}</td>
                      <td style={{ fontWeight: r.key === 'hybrid' ? 700 : 400 }}>
                        {m['recall@10'] ? m['recall@10'].toFixed(4) : '—'}
                      </td>
                      <td>{m['mrr'] ? m['mrr'].toFixed(4) : '—'}</td>
                      <td>{m['ndcg@10'] ? m['ndcg@10'].toFixed(4) : '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          <ul className="note-list">
            {evaluation.notes.map((note, idx) => (
              <li key={idx} className="note-item">
                {note}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  )
}
