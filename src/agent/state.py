"""LangGraph 状态定义：集中管理 Issue RAG Agent 四个节点共享的字段。"""

from typing import List, Optional, TypedDict

from typing_extensions import NotRequired


class IssueState(TypedDict):
    """四个 LangGraph 节点共享的状态，字段按生产者分组。

    `run_agent` 只初始化原始输入和重试控制字段；其余字段由节点逐步写入。
    节点只返回自己更新的字段，LangGraph 会将它们合并进共享状态。
    低置信度重试时，中间结果会被覆盖，`previous_decisions` 则持续追加。
    """

    # run_agent：保存原始输入，并初始化循环控制字段。
    # 例如 raw_issue = "vscode 打开文件时崩溃"。
    raw_issue: str
    retry_count: int
    previous_decisions: NotRequired[List[dict]]

    # query_analysis_node：把原始描述改写成检索输入。
    # 例如 rewritten_query = "vscode file open crash NullPointerException"。
    rewritten_query: str
    keywords: List[str]
    component: Optional[str]

    # retrieval_node：保存 HybridRetriever 返回的 Top-30 候选 Issue。
    retrieved_docs: List[dict]

    # rerank_node：默认保留 Top-5，分数接近时最多扩展到 Top-10。
    reranked_docs: List[dict]

    # decision_node：保存最终判断，并更新 retry_count 和 previous_decisions。
    decision: str
    confidence: float
    related_issues: List[str]
    reasoning: str
