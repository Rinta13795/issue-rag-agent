"""重排序模块：用 CrossEncoder 对 HybridRetriever 返回的候选 issue 做精排。"""

from loguru import logger
from sentence_transformers import CrossEncoder

from config import (
    DYNAMIC_THRESHOLD_DIFF,
    EMBED_DEVICE,
    RERANK_DOC_MAX_CHARS,
    RERANK_MAX_EXTRA_DOCS,
    RERANK_TOP_K,
    RERANKER_MODEL,
)


class Reranker:
    """输入 query 和候选 issue，输出按 CrossEncoder 分数降序排列的候选列表。"""

    def __init__(self) -> None:
        """输入无，加载 config.py 指定的 CrossEncoder 重排序模型。"""
        # CrossEncoder 会让 query 和 doc 在同一个 BERT 中交互，适合对 Top30 候选做精排。
        self.model = CrossEncoder(RERANKER_MODEL, device=EMBED_DEVICE)
        logger.info("Reranker 模型加载完成：{} ({})", RERANKER_MODEL, EMBED_DEVICE)

    def rerank(
        self,
        query: str,
        docs: list[dict],
        top_k: int = RERANK_TOP_K,
        dynamic_threshold: float = DYNAMIC_THRESHOLD_DIFF,
    ) -> list[dict]:
        """输入 query 和候选 issue 列表，输出写入 rerank_score 后的精排结果。"""
        # 没有候选文档时直接返回空列表，避免 CrossEncoder predict 空输入。
        if not docs:
            logger.info("Rerank 跳过：输入 docs 为空")
            return []

        logger.info("Rerank 开始：query={}, docs={}", query, len(docs))

        pairs: list[tuple[str, str]] = []
        for doc in docs:
            # pair 格式是 (query, title + body)，让 CrossEncoder 同时看查询和候选 issue。硬截断方式，此时收益更大
            text = f"{doc.get('title', '')}\n{doc.get('body', '')}"
            #硬截断方式，此时收益更大
            # 截断到 config.py 指定字符数：BERT 最大输入约 512 token，不截断可能报错或被底层不一致截断。
            pairs.append((query, text[:RERANK_DOC_MAX_CHARS]))

        # CrossEncoder 返回分数顺序和 pairs 完全一致，因此可以 zip 回原始 docs。
        scores = self.model.predict(pairs)
        for doc, score in zip(docs, scores):
            # （给每个 doc 写入 rerank_score！！！），后续 Decision 节点可以看到精排分数。
            doc["rerank_score"] = float(score)

        # 按 rerank_score 降序排列，分数越高表示 query-doc 匹配越强。
        sorted_docs = sorted(docs, key=lambda item: -item["rerank_score"])

        # 默认保留 TopK；如果候选不足 TopK，就返回全部候选。
        cut = min(top_k, len(sorted_docs))

        # 动态阈值扩展：TopK 和 TopK+1 分数很接近时，避免武断截断漏掉真正 duplicate。
        max_cut = min(len(sorted_docs), top_k + RERANK_MAX_EXTRA_DOCS)
        while cut < max_cut:
            #当前候选-下一个没加入的候选
            score_gap = sorted_docs[cut - 1]["rerank_score"] - sorted_docs[cut]["rerank_score"]
            if score_gap < dynamic_threshold:
                cut += 1
            else:
                break
            
        #分2种，一种正常情况，文本大的情况就是会向后不断拓展直到达到max_cut。
        # 另外就是文本少的情况，就返回当前文本
        results = sorted_docs[:cut]
        logger.info("Rerank 完成：输入 {} 条，返回 {} 条", len(docs), len(results))
        return results
