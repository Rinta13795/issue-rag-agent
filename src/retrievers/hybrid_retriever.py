"""混合检索模块：调用向量检索和 BM25 检索，并用 RRF 做排名融合。"""

from collections import defaultdict
from typing import Any

from loguru import logger

from config import HYBRID_TOP_K, RRF_K


class HybridRetriever:
    """输入已初始化的向量检索器和 BM25 检索器，输出混合检索器。

    可选传入 docstore（issue_id -> {title, body}）：融合后为缺少正文的候选
    （通常是只被 BM25 召回的 issue）补齐 title/body，让 Reranker 和 Decision
    对所有候选都有文本证据。docstore 为空时行为与 v1 完全一致。
    """

    def __init__(
        self,
        vector_retriever: Any,
        bm25_retriever: Any,
        docstore: dict[str, dict] | None = None,
    ) -> None:
        """输入两个已初始化检索器和可选文档库，保存为混合检索依赖。"""
        # VectorRetriever 负责语义召回，BM25Retriever 负责关键词和错误码等精确召回。
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.docstore = docstore or {}

    def search(
        self,
        query: str,
        top_k: int = HYBRID_TOP_K,
        filter_dict: dict | None = None,
        bm25_extra_terms: list[str] | None = None,
    ) -> list[dict]:
        """输入 query、TopK、可选向量过滤条件和 BM25 补充关键词，输出 RRF 融合结果。"""
        logger.info("混合检索开始：query={}, top_k={}, filter={}", query, top_k, filter_dict)

        # 向量检索可以使用 metadata filter，利用 component 等结构化信息缩小语义召回范围。
        vector_results = self.vector_retriever.search(
            query=query,
            top_k=top_k,
            filter_dict=filter_dict,
        )

        # BM25 不做预过滤，防止 component 判断错误时漏掉关键词强匹配的 duplicate。
        # keywords 作为补充 token 加强错误码、API 名等精确信号的权重。
        bm25_results = self.bm25_retriever.search(
            query=query,
            top_k=top_k,
            extra_terms=bm25_extra_terms,
        )

        # RRF 不做线性加权，因为 BM25 分数无上限、向量相似度约在 0-1，量纲不一致。
        # RRF_K=60 来自 Cormack 2009 论文经验值；按排名加分让两路都靠前的 doc 自然胜出。
        rrf_scores: dict[str, float] = defaultdict(float)
        issue_docs: dict[str, dict] = {}

        # 第一遍融合向量结果：保存完整 issue 信息，并按排名累加 RRF 分。
        for rank, doc in enumerate(vector_results, start=1):
            issue_id = doc["id"]
            rrf_scores[issue_id] += 1.0 / (RRF_K + rank)

            # issue 信息以向量检索结果为准，因为它包含 title/body/metadata。
            issue_docs[issue_id] = {
                "id": issue_id,
                "title": doc.get("title", ""),
                "body": doc.get("body", ""),
                "metadata": doc.get("metadata", {}),
            }

        # 第二遍融合 BM25 结果：BM25 只有 id/score，因此只补最少字段，不覆盖向量路完整信息。
        for rank, doc in enumerate(bm25_results, start=1):
            issue_id = doc["id"]
            rrf_scores[issue_id] += 1.0 / (RRF_K + rank)

            # 只在 BM25 出现的 issue 没有 chunk 内容，先占位，融合后由 docstore 补齐。
            if issue_id not in issue_docs:
                issue_docs[issue_id] = {
                    "id": issue_id,
                    "title": "",
                    "body": "",
                    "metadata": {},
                }

        # 按 RRF 分数降序排序，RRF 共识机制会让两路排名都靠前的 issue 排到更前。
        ranked_ids = sorted(rrf_scores, key=lambda issue_id: -rrf_scores[issue_id])[:top_k]

        # 输出统一格式：score 使用融合后的 RRF 分数，而不是原始 BM25 或向量分数。
        results = []
        for issue_id in ranked_ids:
            result = issue_docs[issue_id].copy()
            result["score"] = rrf_scores[issue_id]
            results.append(result)

        # 融合后统一补齐正文：只被 BM25 召回的候选在这里获得 title/body。
        hydrated = self._hydrate(results)

        logger.info(
            "混合检索完成：vector={} 条，bm25={} 条，返回 {} 个 issue（docstore 补齐 {} 个）",
            len(vector_results),
            len(bm25_results),
            len(results),
            hydrated,
        )
        return results

    def search_queries(
        self,
        queries: list[str],
        top_k: int = HYBRID_TOP_K,
        filter_dict: dict | None = None,
        bm25_extra_terms: list[str] | None = None,
    ) -> list[dict]:
        """分别检索多个 query，再用第二层 RRF 融合为一个候选列表。

        Agent 会按 ``[rewritten_query, raw_issue]`` 的顺序传入两路 query：
        改写结果仍是主要检索表达，原文则在改写语义漂移时提供兜底召回。
        每一路内部仍会执行原有的 Vector + BM25 + RRF，本方法只负责融合
        多个 query 已经排好序的结果，不比较不同检索器的原始分数量纲。
        """
        # 清除空 query，并按忽略大小写后的文本去重。改写失败回退到原文时，
        # 两个 query 完全相同，只执行一次检索，避免无意义地增加一倍耗时。
        unique_queries = []
        seen_queries = set()
        for query in queries:
            normalized = query.strip()
            dedup_key = normalized.casefold()
            if not normalized or dedup_key in seen_queries:
                continue
            seen_queries.add(dedup_key)
            unique_queries.append(normalized)

        if not unique_queries:
            logger.warning("多 Query 检索跳过：没有可用 query")
            return []

        # 只有一个有效 query 时直接返回原有 search 结果，行为与单路检索完全一致。
        if len(unique_queries) == 1:
            return self.search(
                query=unique_queries[0],
                top_k=top_k,
                filter_dict=filter_dict,
                bm25_extra_terms=bm25_extra_terms,
            )

        ranked_lists = [
            self.search(
                query=query,
                top_k=top_k,
                filter_dict=filter_dict,
                bm25_extra_terms=bm25_extra_terms,
            )
            for query in unique_queries
        ]

        # 外层 RRF 只使用“某个 Issue 在各 query 结果中的名次”。
        # 同一 Issue 同时被原文和改写召回时会累加两次，因此排名自然提高。
        rrf_scores: dict[str, float] = defaultdict(float)
        issue_docs: dict[str, dict] = {}
        for docs in ranked_lists:
            for rank, doc in enumerate(docs, start=1):
                issue_id = doc["id"]
                rrf_scores[issue_id] += 1.0 / (RRF_K + rank)

                # 同一 ID 可能在一路只有 BM25 的空正文，在另一路却有 Vector 的完整正文。
                # 保留 title+body 更完整的版本，避免融合后丢掉给 Reranker 的文本证据。
                current = issue_docs.get(issue_id)
                if current is None or self._evidence_length(doc) > self._evidence_length(current):
                    issue_docs[issue_id] = {
                        "id": issue_id,
                        "title": doc.get("title", ""),
                        "body": doc.get("body", ""),
                        "metadata": doc.get("metadata", {}),
                    }

        # Python 排序是稳定的。外层 RRF 分数相同时，先传入的 rewritten_query
        # 结果保持在前，避免原文兜底无依据地压过改写主路。
        ranked_ids = sorted(
            rrf_scores,
            key=lambda issue_id: -rrf_scores[issue_id],
        )[:top_k]

        results = []
        for issue_id in ranked_ids:
            result = issue_docs[issue_id].copy()
            result["score"] = rrf_scores[issue_id]
            results.append(result)

        # 内层 search 已各自补齐过正文，这里再补一次覆盖跨路融合后的遗漏，幂等无副作用。
        self._hydrate(results)

        logger.info(
            "多 Query 检索完成：queries={}，各路返回={}，融合后={} 个 issue",
            len(unique_queries),
            [len(docs) for docs in ranked_lists],
            len(results),
        )
        return results

    def _hydrate(self, results: list[dict]) -> int:
        """为缺少 title/body 的候选从 docstore 补齐正文，返回补齐数量。

        只填空缺、不覆盖：向量路候选的 body 是与 query 最相关的 chunk，
        对 Reranker 是比“正文开头 N 字符”更好的证据，保留原值。
        """
        if not self.docstore:
            return 0

        hydrated = 0
        for doc in results:
            record = self.docstore.get(str(doc["id"]))
            if record is None:
                continue

            changed = False
            if not str(doc.get("title", "")).strip() and record.get("title"):
                doc["title"] = record["title"]
                changed = True
            if not str(doc.get("body", "")).strip() and record.get("body"):
                doc["body"] = record["body"]
                changed = True
            if changed:
                hydrated += 1
        return hydrated

    @staticmethod
    def _evidence_length(doc: dict) -> int:
        """用 title 与 body 的非空字符数衡量候选携带的文本证据量。"""
        title = str(doc.get("title", "")).strip()
        body = str(doc.get("body", "")).strip()
        return len(title) + len(body)
