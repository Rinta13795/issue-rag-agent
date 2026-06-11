"""LangGraph 状态定义：集中管理 Issue RAG Agent 四个节点共享的字段。"""

from typing import List, Optional, TypedDict

from typing_extensions import NotRequired


class IssueState(TypedDict):
    """输入输出为 LangGraph 节点共享状态；每个节点只返回自己更新的字段。"""

    raw_issue: str
    rewritten_query: str
    keywords: List[str]
    component: Optional[str]
    retrieved_docs: List[dict]
    reranked_docs: List[dict]
    decision: str
    confidence: float
    related_issues: List[str]
    reasoning: str
    retry_count: int
    previous_decisions: NotRequired[List[dict]]


"""
  ---
  好问题，这正是 LangGraph 的核心价值。如果只是顺序传递，直接函数套函数就行：

  query = query_analysis(raw_issue)
  docs = retrieval(query)
  reranked = rerank(query, docs)
  result = decision(reranked)

  这样写完全能跑。那为什么还要 state？三个原因：
  
  ---
  原因1：你的流程不是直线，是带回环的
  
  decision 低置信度时要回到 query_analysis 重写 query，函数链没法跳回去。你得自己手写：

  while True:
      query = query_analysis(raw_issue, previous_decisions)
      docs = retrieval(query, component)
      reranked = rerank(query, docs)
      decision, conf, history = decision(reranked, previous_decisions)
      if conf >= 0.7 or retry > 2:
          break
      previous_decisions = history
      retry += 1
  
  这就是你自己手写一个简陋版 state 机。LangGraph 把这套循环+状态管理标准化了。
  
  ---
  原因2：跨节点共享，不只是相邻传递
  
  query_analysis 算出的 component 要传给 retrieval，但重试时 query_analysis 还要读
  decision 产出的 previous_decisions——隔了三个节点。函数链里只能层层透传，state
  是共享盒子，谁都能直接读任意字段。

  ---
  原因3：节点只返回自己改的那部分
  
  每个节点只返回 {"retrieved_docs": docs}，不是整个 state。LangGraph
  自动合并进总状态，这叫状态合并（reducer）。
  
  ---
  面试一句话：
  
  ▎ 简单线性 pipeline 用函数链够了。但我的 Agent 
  ▎ 有条件循环（低置信度重写）和跨节点状态共享（重试时读历史决策），LangGraph 的 state + 
  ▎ 条件边把"循环控制 + 状态合并"标准化了，不用手写 while 循环和参数透传。


"""


"""
  ---
  run_agent 初始化时填入：
  - raw_issue — 原始 issue 文本，比如 "vscode打开文件崩溃"
  - retry_count — 初始值 0
  - previous_decisions — 初始值 []
  
  ---
  query_analysis_node 填入：
  - rewritten_query — 改写后的技术 query，比如 "vscode file open crash NullPointerException"
  - keywords — 提取的关键词，比如 ["vscode", "crash", "file"]
  - component — 识别出的组件，比如 "vscode"
  
  ---
  retrieval_node 填入：
  - retrieved_docs — HybridRetriever 返回的 Top30 候选 issue

  ---
  rerank_node 填入：
  - reranked_docs — Reranker 精排后的 Top5 候选 issue

  ---
  decision_node 填入：
  - decision — 判断结果，"duplicate" / "similar" / "new"
  - confidence — 置信度，比如 0.85
  - related_issues — 相关 issue 的 id 列表
  - reasoning — 判断理由
  - retry_count — 累加 +1
  - previous_decisions — 追加本轮历史


"""