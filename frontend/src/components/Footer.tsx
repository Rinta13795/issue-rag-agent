import React from 'react'

export const Footer: React.FC = () => {
  return (
    <footer className="app-footer">
      <div className="footer-mark">
        <span className="square">■</span>
        <span>ISSUE INTELLIGENCE · 研发问题分诊 · 证据辅助决策</span>
      </div>

      <div style={{ display: 'flex', gap: '16px' }}>
        <a
          href="https://github.com/Rinta13795/issue-rag-agent"
          target="_blank"
          rel="noreferrer"
          className="footer-link"
        >
          项目源码 →
        </a>
      </div>
    </footer>
  )
}
