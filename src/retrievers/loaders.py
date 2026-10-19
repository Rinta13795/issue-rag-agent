"""检索依赖加载器：混合检索器和重排模型，进程内只创建一次。

聊天（search_issues 工具）、检索评测和旧 LangGraph 流水线都从这里拿，
互相不再依赖对方的文件。重模型的 import 都放在函数里，导入本模块不会加载模型。
"""

import threading

from loguru import logger

from config import CHROMA_COLLECTION, CHROMA_PERSIST_DIR

_HYBRID_RETRIEVER = None
_RERANKER = None
_LOCK = threading.Lock()


def get_hybrid_retriever():
    """输入无，输出已初始化的 HybridRetriever，供 Retrieval 节点复用。"""
    # Chroma 和检索器依赖只在真正构建 Agent 图时导入，避免 import run_agent 时提前加载重模型。
    from langchain_chroma import Chroma

    from src.docstore import load_docstore
    from src.indexer import load_embeddings
    from src.retrievers.bm25_retriever import BM25Retriever
    from src.retrievers.hybrid_retriever import HybridRetriever
    from src.retrievers.vector_retriever import VectorRetriever

    # ChromaDB 读取 Step2 构建好的持久化向量库，Embedding 用同一套本地模型。
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=load_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )

    # VectorRetriever 负责语义召回，BM25Retriever 负责关键词召回，HybridRetriever 用 RRF 融合。
    # docstore 为融合后缺正文的候选（BM25-only）补齐 title/body；文件缺失时退回 v1 行为。
    vector_retriever = VectorRetriever(vectorstore)
    bm25_retriever = BM25Retriever()
    return HybridRetriever(vector_retriever, bm25_retriever, docstore=load_docstore())


def get_reranker():
    """输入无，输出已初始化的 Reranker，供 Rerank 节点复用。"""
    # Reranker 内部会加载 CrossEncoder，延迟到 run_agent/build_graph 时再加载，避免 import 阶段失败。
    from src.reranker import Reranker

    return Reranker()


def get_cached_reranker():
    """只要重排模型时用：GitHub 仓库快照有自己的检索器，不必加载十万条 GitBugs 的 BM25 索引。"""
    global _RERANKER
    if _RERANKER is None:
        with _LOCK:
            if _RERANKER is None:
                _RERANKER = get_reranker()
    return _RERANKER


def get_retrieval_dependencies():
    """返回缓存的 GitBugs 混合检索器与重排模型，首次调用时才加载。"""
    global _HYBRID_RETRIEVER
    if _HYBRID_RETRIEVER is None:
        with _LOCK:
            if _HYBRID_RETRIEVER is None:
                logger.info("初始化检索依赖：HybridRetriever")
                _HYBRID_RETRIEVER = get_hybrid_retriever()
    return _HYBRID_RETRIEVER, get_cached_reranker()
