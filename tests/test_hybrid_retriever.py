"""HybridRetriever 多 Query 融合的快速单元测试。"""

import sys
import types
import unittest

# 这组测试只验证排名融合，不需要真实日志和 .env。允许在未安装完整项目依赖的
# 轻量环境中运行；正式环境已安装模块时仍使用真实依赖。
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
    def __init__(self, results_by_query: dict[str, list[dict]]) -> None:
        self.results_by_query = results_by_query
        self.calls: list[str] = []

    def search(
        self,
        query: str,
        top_k: int,
        filter_dict: dict | None = None,
    ) -> list[dict]:
        self.calls.append(query)
        return [doc.copy() for doc in self.results_by_query.get(query, [])][:top_k]


class FakeBM25Retriever:
    def __init__(self, results_by_query: dict[str, list[dict]] | None = None) -> None:
        self.results_by_query = results_by_query or {}
        self.calls: list[str] = []

    def search(self, query: str, top_k: int) -> list[dict]:
        self.calls.append(query)
        return [doc.copy() for doc in self.results_by_query.get(query, [])][:top_k]


def _doc(issue_id: str, body: str) -> dict:
    return {
        "id": issue_id,
        "title": issue_id,
        "body": body,
        "metadata": {},
        "score": 1.0,
    }


class HybridRetrieverMultiQueryTest(unittest.TestCase):
    def test_fuses_rankings_and_keeps_more_complete_evidence(self) -> None:
        vector = FakeVectorRetriever(
            {
                "rewritten": [_doc("a", "short"), _doc("b", "rewritten only")],
                "raw": [_doc("c", "raw only"), _doc("a", "much longer evidence")],
            }
        )
        retriever = HybridRetriever(vector, FakeBM25Retriever())

        results = retriever.search_queries(["rewritten", "raw"], top_k=3)

        self.assertEqual([doc["id"] for doc in results], ["a", "c", "b"])
        self.assertEqual(results[0]["body"], "much longer evidence")
        self.assertGreater(results[0]["score"], results[1]["score"])
        self.assertGreater(results[1]["score"], results[2]["score"])

    def test_deduplicates_equivalent_queries(self) -> None:
        vector = FakeVectorRetriever({"Crash": [_doc("a", "evidence")]})
        bm25 = FakeBM25Retriever()
        retriever = HybridRetriever(vector, bm25)

        results = retriever.search_queries([" Crash ", "crash"], top_k=3)

        self.assertEqual([doc["id"] for doc in results], ["a"])
        self.assertEqual(vector.calls, ["Crash"])
        self.assertEqual(bm25.calls, ["Crash"])

    def test_returns_empty_for_blank_queries(self) -> None:
        vector = FakeVectorRetriever({})
        bm25 = FakeBM25Retriever()
        retriever = HybridRetriever(vector, bm25)

        self.assertEqual(retriever.search_queries([" ", ""], top_k=3), [])
        self.assertEqual(vector.calls, [])
        self.assertEqual(bm25.calls, [])


if __name__ == "__main__":
    unittest.main()
