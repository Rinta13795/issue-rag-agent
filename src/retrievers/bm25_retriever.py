"""BM25 检索模块：加载 issue 粒度 BM25 索引，并按关键词相关性返回 issue id。"""

import pickle
from pathlib import Path
from typing import Any

from loguru import logger

from config import BM25_INDEX_PATH, BM25_MIN_SCORE, BM25_TOP_K
from src.indexer import tokenize


class BM25Retriever:
    """输入 BM25 索引文件路径，输出可按 query 检索 issue id 的关键词检索器。"""

    def __init__(self, bm25_path: str = BM25_INDEX_PATH) -> None:
        """输入 bm25.pkl 路径，一次性加载 bm25 对象和 issue id 列表。"""
        # BM25 索引在初始化时加载一次，避免每次 search 都重复读 pickle 文件。
        index_path = Path(bm25_path)
        logger.info("加载 BM25 索引：{}", index_path)
        with index_path.open("rb") as file:
            data: dict[str, Any] = pickle.load(file)

        # bm25 是 BM25Okapi 对象，ids 的下标顺序必须和 bm25 corpus 的下标顺序一致。————存着各种索引模块
        self.bm25 = data["bm25"]
        self.ids: list[str] = data["ids"]

    def search(
        self,
        query: str,
        top_k: int = BM25_TOP_K,
        extra_terms: list[str] | None = None,
    ) -> list[dict]:
        """输入 query、TopK 和可选补充关键词，输出按 BM25 分数降序排列的 issue id 与 score 列表。

        extra_terms 是 Query Analysis 提取的 keywords：追加到 query token 中相当于
        提升错误码、API 名等精确信号的词频权重。v1 中 keywords 只在 Decision prompt
        中展示，没有参与任何检索，这里让它真正作用于 BM25。
        """
        # 查询分词必须复用建索引时的 tokenize，保证 jieba + 空格细分 + 小写规则完全一致。
        query_tokens = tokenize(query)

        # keywords 同样过 tokenize，保证和索引词汇的切分、小写规则一致。
        for term in extra_terms or []:
            if isinstance(term, str) and term.strip():
                query_tokens.extend(tokenize(term))

        logger.info("BM25 检索开始：query={}, top_k={}, extra_terms={}", query, top_k, extra_terms)

        # BM25 是 issue 粒度索引，get_scores 返回全量 issue 分数，不需要 chunk 聚合。
        scores = self.bm25.get_scores(query_tokens)

        # scores 下标和 self.ids 下标一一对应，按分数降序取 TopK 下标。
        top_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        # 过滤零分候选：BM25 为 0 表示 query 词一个都没命中，这类候选进入 RRF
        # 只会挤占真实候选的排名配额（v1 已知限制之一）。
        results = [
            {"id": self.ids[index], "score": float(scores[index])}
            for index in top_indices
            if float(scores[index]) > BM25_MIN_SCORE
        ]

        logger.info("BM25 检索完成：返回 {} 个 issue", len(results))
        return results
"""
  BM25 的倒排索引存的是词 → 哪些 issue 包含这个词。

  查询时必须先切词，才能去索引里查：
  - 查 "保存" → 哪些 issue 有这个词
  - 查 "崩溃" → 哪些 issue 有这个词
  - 查 "typeerror" → 哪些 issue 有这个词          

"""