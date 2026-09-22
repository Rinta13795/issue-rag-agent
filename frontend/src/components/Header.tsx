import React from 'react'
import type { SystemInfo } from '../types'

interface HeaderProps {
  systemInfo: SystemInfo | null
}

export const Header: React.FC<HeaderProps> = ({ systemInfo }) => (
  <header className="masthead">
    <div className="masthead-pre">ISSUE TRIAGE / INTELLIGENT WORKSPACE</div>
    <h1 className="masthead-title">让 Agent 接手问题初筛<span className="dot">。</span></h1>
    <p className="masthead-sub">提交一个问题，交付一份有依据的分诊建议。</p>
    <p className="masthead-tech">从历史 Issue 中查找关联线索，完成重复识别、证据核验与处理建议。</p>
    <dl className="workspace-facts">
      <div><dt>历史问题库</dt><dd>{systemInfo ? systemInfo.dataset_size.toLocaleString() : '待连接'}<span> 条 Issue</span></dd></div>
      <div><dt>分析范围</dt><dd>重复识别与关联排查</dd></div>
      <div><dt>交付内容</dt><dd>分诊建议 + 原文证据</dd></div>
    </dl>
    <div className="masthead-rule" />
  </header>
)
