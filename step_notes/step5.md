# Step 5: LangGraph 状态机编排

## 这一步做什么

用 LangGraph 把 Query Analysis、Retrieval、Rerank、Decision 四个节点串成状态机，并通过条件边实现低置信度重写循环。

## 涉及的文件

- `src/agent/state.py`
- `src/agent/nodes.py`
- `src/agent/graph.py`
- `src/agent/__init__.py`

## 输入输出

- 输入：`issue_text: str`，即用户原始 issue title + body。
- 输出：`{decision, confidence, related_issues, reasoning, retry_count}`。

## 关键设计细节

状态对象：

```python
from typing import TypedDict, List, Optional
from typing_extensions import NotRequired

class IssueState(TypedDict):
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
```

四个节点函数：`query_analysis_node(state) -> dict` 输出 query 改写结果；`retrieval_node(state) -> dict` 输出 `retrieved_docs`；`rerank_node(state) -> dict` 输出 `reranked_docs`；`decision_node(state) -> dict` 输出最终判断和循环历史。

Query Analysis 重试时读取 `previous_decisions` 最新一轮，把低置信度和 Top3 检索结果标题拼入 `{feedback}`，让 LLM 基于上一轮失败线索重写 query。

Retrieval 节点使用 `{"component": state["component"]}` 构造 metadata filter，调用 `retriever.search(query=state["rewritten_query"], top_k=HYBRID_TOP_K, filter_dict=filter_dict)`，返回 `{"retrieved_docs": docs}`。

Decision 节点维护 `previous_decisions`，记录 `rewritten_query`、`confidence`、`reranked_docs`、`decision`，并返回 `decision`、`confidence`、`related_issues`、`reasoning`、`retry_count + 1`。

条件边：

```python
def should_retry(state: IssueState) -> str:
    if state["confidence"] < CONFIDENCE_THRESHOLD and state["retry_count"] <= MAX_RETRIES:
        log.warning(f"Retry triggered: conf={state['confidence']:.2f}, count={state['retry_count']}")
        return "retry"
    return "end"
```

图结构：

```python
def build_graph():
    g = StateGraph(IssueState)
    g.add_node("query_analysis", query_analysis_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("rerank", rerank_node)
    g.add_node("decision", decision_node)
    g.set_entry_point("query_analysis")
    g.add_edge("query_analysis", "retrieval")
    g.add_edge("retrieval", "rerank")
    g.add_edge("rerank", "decision")
    g.add_conditional_edges("decision", should_retry, {"retry": "query_analysis", "end": END})
    return g.compile()
```

入口函数 `run_agent(issue_text: str) -> dict`：初始化 `{"raw_issue": issue_text, "retry_count": 0, "previous_decisions": []}`，调用 graph，返回 `decision`、`confidence`、`related_issues`、`reasoning`、`retry_count`。

## 关键参数

- `HYBRID_TOP_K`
- `RERANK_TOP_K`
- `DYNAMIC_THRESHOLD_DIFF`
- `CONFIDENCE_THRESHOLD`
- `MAX_RETRIES`

## 依赖哪些已完成的模块

- Step 1：全局配置和日志。
- Step 3：`get_hybrid_retriever()` 需要混合检索。
- Step 4：`get_reranker()` 需要重排序模块。
- Step 6：节点 Prompt 和 `format_candidates`。

## 完成标志

- [ ] `IssueState` 字段完整。
- [ ] 四个节点函数实现并返回局部 state dict。
- [ ] `should_retry` 按置信度和重试次数判断。
- [ ] `build_graph` 添加四节点、三条普通边、一条条件边。
- [ ] `run_agent(issue_text)` 返回最终决策字典。

## 踩过的坑
