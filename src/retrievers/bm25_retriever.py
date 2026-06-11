"""BM25 检索模块：加载 issue 粒度 BM25 索引，并按关键词相关性返回 issue id。"""

import pickle
from pathlib import Path
from typing import Any

from loguru import logger

from config import BM25_INDEX_PATH, BM25_TOP_K
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

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[dict]:
        """输入 query 和 TopK，输出按 BM25 分数降序排列的 issue id 与 score 列表。"""
        # 查询分词必须复用建索引时的 tokenize，保证 jieba + 空格细分 + 小写规则完全一致。————将词分开
        query_tokens = tokenize(query)
        logger.info("BM25 检索开始：query={}, top_k={}", query, top_k)

        # BM25 是 issue 粒度索引，get_scores 返回全量 issue 分数，不需要 chunk 聚合。
        scores = self.bm25.get_scores(query_tokens)

        # scores 下标和 self.ids 下标一一对应，按分数降序取 TopK 下标。
        top_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],#按照下标对应大小分数，对下标进行排序
            reverse=True,
        )[:top_k]#表示取前几个

        # 通过 ids[下标] 找回 issue_id，只返回 id 和 score，不返回 chunk 内容。
        # ——————从top score向下一步一步返回，id和score通过index一一对应
        results = [
            {"id": self.ids[index], "score": float(scores[index])}
            for index in top_indices
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