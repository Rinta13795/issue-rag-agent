"""Issue 文档库：保存 issue_id -> {title, body} 映射，为检索候选补齐正文证据。

v1 的已知限制：只被 BM25 召回的候选没有 title/body，Reranker 和 Decision 拿不到
它们的文本证据。本模块在离线阶段把清洗后的 issue 正文（截断）持久化，在线阶段
按 id 补齐空正文候选，让所有候选都能参与精排和决策。
"""

import pickle
from pathlib import Path

from loguru import logger

from config import DOCSTORE_BODY_MAX_CHARS, DOCSTORE_PATH


def build_docstore(
    issues: list[dict],
    path: str = DOCSTORE_PATH,
    body_max_chars: int = DOCSTORE_BODY_MAX_CHARS,
) -> dict[str, dict]:
    """输入清洗后的标准 issue 列表，构建并持久化 id -> {title, body} 文档库。

    body 截断到 body_max_chars：Reranker 只看前 800 字符、Decision 只看前几百字符，
    保存完整正文只会让文件变大而没有下游收益。
    """
    store = {
        str(issue["id"]): {
            "title": str(issue.get("title", "")),
            "body": str(issue.get("body", ""))[:body_max_chars],
        }
        for issue in issues
    }

    output_path = Path(path)
    with output_path.open("wb") as file:
        pickle.dump(store, file)

    logger.info("docstore 构建完成：{} 条 issue -> {}", len(store), output_path)
    return store


def load_docstore(path: str = DOCSTORE_PATH) -> dict[str, dict]:
    """输入 docstore 文件路径，输出 id -> {title, body} 字典。文件不存在时返回空字典。

    返回空字典而不是抛异常：docstore 是证据增强，不是检索的硬依赖；
    缺失时系统退回 v1 行为（BM25-only 候选无正文），只损失效果不损失可用性。
    """
    store_path = Path(path)
    if not store_path.exists():
        logger.warning("docstore 文件不存在：{}，BM25-only 候选将没有正文证据", store_path)
        return {}

    with store_path.open("rb") as file:
        store = pickle.load(file)

    logger.info("docstore 加载完成：{} 条 issue", len(store))
    return store
