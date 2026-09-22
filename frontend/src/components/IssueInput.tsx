import React, { useState } from 'react'
import type { ExampleItem } from '../types'

interface IssueInputProps {
  examples: ExampleItem[]
  onAnalyze: (text: string, sampleId?: string) => void
  isAnalyzing: boolean
}

export const IssueInput: React.FC<IssueInputProps> = ({ examples, onAnalyze, isAnalyzing }) => {
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [selectedExampleId, setSelectedExampleId] = useState<string | null>(null)

  const handleSelectExample = (ex: ExampleItem) => {
    setSelectedExampleId(ex.id)
    setTitle(ex.title)
    setDescription(ex.description)
  }

  const handleClear = () => {
    setTitle('')
    setDescription('')
    setSelectedExampleId(null)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const fullText = title.trim() ? `${title.trim()}\n${description.trim()}` : description.trim()
    if (!fullText) return
    onAnalyze(fullText, selectedExampleId || undefined)
  }

  return (
    <section className="block intake-panel">
      <div className="block-label">
        <span className="cn">给 Agent 下达任务</span>
        <span className="en">NEW TASK</span>
      </div>

      <div style={{ marginBottom: '16px' }}>
        <div className="field-label" style={{ marginBottom: '8px' }}>
          快速体验 · 选择典型问题载入
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {examples.map((ex) => (
            <button
              key={ex.id}
              type="button"
              className={`btn btn--sm ${selectedExampleId === ex.id ? 'btn--primary' : ''}`}
              onClick={() => handleSelectExample(ex)}
              disabled={isAnalyzing}
              title={ex.note}
            >
              <span>{ex.label}</span>
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="field-group">
          <label className="field-label" htmlFor="issue-title">问题标题（选填）</label>
          <input
            id="issue-title"
            type="text"
            className="field-input"
            placeholder="例如：应用启动时出现 NullPointerException"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            disabled={isAnalyzing}
          />
        </div>

        <div className="field-group">
          <label className="field-label" htmlFor="issue-description">问题描述与复现步骤（必填）</label>
          <textarea
            id="issue-description"
            className="field-textarea"
            placeholder="描述问题现象、影响范围与复现步骤，可附上错误日志或堆栈信息。"
            rows={5}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            disabled={isAnalyzing}
            required
          />
        </div>

        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <button
            type="submit"
            className="btn btn--primary"
            disabled={isAnalyzing || (!title.trim() && !description.trim())}
          >
            {isAnalyzing ? '正在分析…' : '启动 Agent →'}
          </button>

          <button
            type="button"
            className="btn"
            onClick={handleClear}
            disabled={isAnalyzing || (!title && !description)}
          >
            清空内容
          </button>
        </div>
      </form>
    </section>
  )
}
