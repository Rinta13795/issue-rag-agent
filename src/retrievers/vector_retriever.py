"""向量检索模块：基于已初始化的 ChromaDB 实例做 chunk 检索，并聚合到 issue 粒度。"""

from typing import Any

from loguru import logger

from config import VECTOR_CHUNK_FETCH_MULTIPLIER, VECTOR_TOP_K


class VectorRetriever:
    """输入外部初始化好的 Chroma vectorstore，输出可按 query 检索 issue 的向量检索器。"""

    def __init__(self, vectorstore: Any) -> None:
        """输入已初始化的 langchain_chroma.Chroma 实例，保存为检索器内部依赖。"""
        # vectorstore 由调用方负责初始化，当前类不加载 embedding，也不创建 ChromaDB。
        self.vs = vectorstore

    def search(
        self,
        query: str,
        top_k: int = VECTOR_TOP_K,
        filter_dict: dict | None = None,
    ) -> list[dict]:
        """输入 query、TopK 和可选 metadata filter，输出按相似度排序的 issue 级结果列表。"""
        # 记录本次向量检索请求，便于排查 query 和 metadata filter 是否符合预期。
        logger.info("向量检索开始：query={}, top_k={}, filter={}", query, top_k, filter_dict)

        # ChromaDB 按 chunk 粒度存储，因此先多召回 chunk，再聚合到 issue 粒度。
        chunk_results = self.vs.similarity_search_with_score(
            query,
            k=top_k * VECTOR_CHUNK_FETCH_MULTIPLIER,
            filter=filter_dict,
        )

        issue_best: dict[str, dict] = {}
        for doc, distance in chunk_results:
            # 每个 chunk 的 metadata 必须包含 issue_id，用它把多个 chunk 聚合回同一个 issue。
            issue_id = doc.metadata.get("issue_id")
            if not issue_id:
                continue

            # Chroma 返回 cosine distance，越小越相似；这里转成 similarity，越大越相关。
            score = 1 - float(distance)

            # 同一 issue 只保留相似度最高的 chunk，body 用该 chunk 的 page_content。
            if issue_id not in issue_best or score > issue_best[issue_id]["score"]:
                issue_best[issue_id] = {
                    "id": issue_id,
                    "title": doc.metadata.get("title", ""),
                    "body": doc.page_content,
                    "score": score,
                    "metadata": dict(doc.metadata),
                }

        # 聚合后按 issue 级相似度降序排序，并截断到 top_k 条。————不理解，就先这样
        results = sorted(issue_best.values(), key=lambda item: -item["score"])[:top_k]

        # 记录最终返回的 issue 数量，便于观察 chunk 聚合后的召回情况。
        logger.info("向量检索完成：返回 {} 个 issue", len(results))
        return results

"""
  ▎ "ChromaDB 按 chunk 粒度存储和召回，但下游需要 issue 粒度。我多召回一倍 chunk 后聚合到 
  ▎ issue，同一 issue 取最高分的 chunk 作为代表，避免下游看到重复 issue。"
"""