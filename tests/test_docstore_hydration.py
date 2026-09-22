"""HybridRetriever docstore 补齐与 BM25 零分过滤的单元测试。"""

import sys
import types
import unittest

# 允许在未安装完整依赖的轻量环境运行：与既有测试保持同一兜底方式。
try:
    import loguru  # noqa: F401
except ModuleNotFoundError:
    class _SilentLogger:
        def info(self, *args: object, **kwargs: object) -> None:
            pass

        def warning(self, *args: object, **kwargs: object) -> None:
            pass

    sys.modules["loguru"] = types.SimpleNamespace(logger=_SilentLogger())

try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda: None)

from src.retrievers.hybrid_retriever import HybridRetriever


class FakeVectorRetriever:
    """固定返回预设结果的向量检索器替身。"""

    def __init__(self, results: list[dict]) -> None:
        self.results = results

    def search(self, query: str, top_k: int, filter_dict: dict | None = None) -> list[dict]:
        return [doc.copy() for doc in self.results[:top_k]]


class FakeBM25Retriever:
    """固定返回预设结果的 BM25 检索器替身，记录收到的 extra_terms。"""

    def __init__(self, results: list[dict]) -> None:
        self.results = results
        self.seen_extra_terms: list[list[str] | None] = []

    def search(
        self,
        query: str,
        top_k: int,
        extra_terms: list[str] | None = None,
    ) -> list[dict]:
        self.seen_extra_terms.append(extra_terms)
        return [doc.copy() for doc in self.results[:top_k]]


class DocstoreHydrationTest(unittest.TestCase):
    def test_bm25_only_candidate_gets_title_and_body(self) -> None:
        """只被 BM25 召回的候选应从 docstore 补齐正文（v1 已知限制）。"""
        vector = FakeVectorRetriever(
            [{"id": "v1", "title": "vector title", "body": "vector body", "metadata": {}}]
        )
        bm25 = FakeBM25Retriever([{"id": "b1", "score": 3.2}])
        docstore = {
            "b1": {"title": "bm25 issue title", "body": "bm25 issue body"},
            "v1": {"title": "SHOULD NOT OVERWRITE", "body": "SHOULD NOT OVERWRITE"},
        }
        retriever = HybridRetriever(vector, bm25, docstore=docstore)

        results = retriever.search("query", top_k=10)
        by_id = {doc["id"]: doc for doc in results}

        self.assertEqual(by_id["b1"]["title"], "bm25 issue title")
        self.assertEqual(by_id["b1"]["body"], "bm25 issue body")
        # 向量路候选的 body 是最相关 chunk，不能被 docstore 的正文开头覆盖。
        self.assertEqual(by_id["v1"]["title"], "vector title")
        self.assertEqual(by_id["v1"]["body"], "vector body")

    def test_without_docstore_behaves_like_v1(self) -> None:
        vector = FakeVectorRetriever([])
        bm25 = FakeBM25Retriever([{"id": "b1", "score": 1.0}])
        retriever = HybridRetriever(vector, bm25)

        results = retriever.search("query", top_k=10)
        self.assertEqual(results[0]["title"], "")
        self.assertEqual(results[0]["body"], "")

    def test_search_queries_hydrates_after_outer_fusion(self) -> None:
        vector = FakeVectorRetriever([])
        bm25 = FakeBM25Retriever([{"id": "b1", "score": 2.0}])
        docstore = {"b1": {"title": "t", "body": "b"}}
        retriever = HybridRetriever(vector, bm25, docstore=docstore)

        results = retriever.search_queries(["query one", "query two"], top_k=10)
        self.assertEqual(results[0]["title"], "t")
        self.assertEqual(results[0]["body"], "b")

    def test_extra_terms_forwarded_to_bm25(self) -> None:
        """keywords 应穿透 HybridRetriever 传给 BM25（v1 中 keywords 不参与检索）。"""
        vector = FakeVectorRetriever([])
        bm25 = FakeBM25Retriever([])
        retriever = HybridRetriever(vector, bm25)

        retriever.search("query", top_k=5, bm25_extra_terms=["NullPointerException"])
        self.assertEqual(bm25.seen_extra_terms, [["NullPointerException"]])


if __name__ == "__main__":
    unittest.main()
