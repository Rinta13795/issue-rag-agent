"""LangGraph 状态机：组装四个节点、条件边和 run_agent 入口。将node逻辑都穿起来"""

from langgraph.graph import END, StateGraph
from loguru import logger
import threading

from config import CONFIDENCE_THRESHOLD, MAX_RETRIES
from src.agent.nodes import (
    configure_dependencies,
    decision_node,
    query_analysis_node,
    rerank_node,
    retrieval_node,
)
from src.agent.state import IssueState
from src.retrievers.loaders import get_retrieval_dependencies


_DEPENDENCIES_READY = False
_COMPILED_GRAPH = None
_DEPENDENCIES_LOCK = threading.Lock()


def _ensure_dependencies() -> None:
    """确保旧流水线的节点拿到检索器和重排模型；加载本身交给 loaders，只做一次。"""
    global _DEPENDENCIES_READY
    if _DEPENDENCIES_READY:
        return
    with _DEPENDENCIES_LOCK:
        if _DEPENDENCIES_READY:
            return
        logger.info("初始化 LangGraph 依赖：HybridRetriever + Reranker")
        retriever, reranker = get_retrieval_dependencies()
        configure_dependencies(retriever, reranker)
        _DEPENDENCIES_READY = True


# build_graph的辅助函数
def should_retry(state: IssueState) -> str:
    """输入当前 IssueState，输出条件边名称 retry 或 end。"""
    # 置信度低且未超过最大重试次数时，回到 Query Analysis 节点重写 query。
    if state["confidence"] < CONFIDENCE_THRESHOLD and state["retry_count"] <= MAX_RETRIES:
        logger.warning(
            "触发 retry：confidence={}, retry_count={}",
            state["confidence"],
            state["retry_count"],
        )
        return "retry"

    logger.info("不触发 retry：进入 END")
    return "end"


def build_graph():
    """输入无，输出编译后的 LangGraph 状态机。"""
    # 构图前先确保依赖已注入到 nodes.py，但整个进程只初始化一次。
    _ensure_dependencies()

    # 用 IssueState 作为四个节点共享状态类型，LangGraph 会自动合并节点返回的局部 dict。
    graph = StateGraph(IssueState)

    # 添加四个核心节点：Query Analysis -> Retrieval -> Rerank -> Decision。
    graph.add_node("query_analysis", query_analysis_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("decision", decision_node)

    # 固定顺序边：先改写 query，再检索，再重排，最后决策。
    graph.set_entry_point("query_analysis")
    graph.add_edge("query_analysis", "retrieval")
    graph.add_edge("retrieval", "rerank")
    graph.add_edge("rerank", "decision")

    # 条件边：Decision 后低置信度回 Query Analysis，否则结束。
    graph.add_conditional_edges(
        "decision",
        # 判断函数
        should_retry,
        {"retry": "query_analysis", "end": END},
    )
    """
  decision 跑完
      → should_retry(state)
          → confidence < 0.7 且 retry <= 2 → 返回 "retry" → 去 query_analysis
          → 否则 → 返回 "end" → 结束
    """
    # 搭建状态机
    return graph.compile()

def get_graph():
    """输入无，输出进程内唯一的已编译 LangGraph 实例。

    v1 每次 run_agent 都重新 build_graph（重依赖有缓存，但图对象每次重建）；
    这里把编译结果也缓存，反复调用 run_agent 时不再有构图开销。
    """
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_graph()
    return _COMPILED_GRAPH


def run_agent(issue_text: str) -> dict:
    """输入原始 issue 文本，输出最终决策结果。"""
    logger.info("开始运行 Issue RAG Agent")
    graph = get_graph()

    # 初始化 LangGraph 状态，只放原始 issue 和循环控制字段，其余字段由节点逐步写入。
    initial_state = {
        "raw_issue": issue_text,
        "retry_count": 0,
        "previous_decisions": [],
    }
    # 开始使用，final就是结果
    """
    graph.invoke(initial_state)获得的LangGraph去执行逻辑
    invoke每一个节点执行逻辑的时候，按节点执行顺序，每个节点 return 的 dict 用 dict.update 合并进容器：
    - 新 key → 加进去
    - 旧 key → 覆盖
    后面节点拿到的就是被前面节点更新过的容器。
    """
    final = graph.invoke(initial_state)


    # 只返回对外需要的最终决策字段。
    result = {
        "decision": final["decision"],
        "confidence": final["confidence"],
        "related_issues": final["related_issues"],
        "reasoning": final["reasoning"],
        "retry_count": final["retry_count"],
    }
    logger.info("Issue RAG Agent 运行完成：{}", result)
    return result
