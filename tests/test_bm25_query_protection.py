"""测试 BM25 查询长度保护与 Extra terms 追加机制。"""

import unittest
from unittest.mock import MagicMock, patch

from config import BM25_MAX_QUERY_CHARS
from src.retrievers.bm25_retriever import BM25Retriever


class BM25QueryProtectionTest(unittest.TestCase):
    @patch("src.retrievers.bm25_retriever.pickle.load")
    @patch("src.retrievers.bm25_retriever.Path.open")
    def test_long_query_is_truncated(self, mock_open, mock_pickle):
        mock_bm25 = MagicMock()
        mock_bm25.get_scores.return_value = [1.5, 0.0]
        mock_pickle.return_value = {"bm25": mock_bm25, "ids": ["id_1", "id_2"]}

        retriever = BM25Retriever("fake_path.pkl")

        # 构造一个超过 BM25_MAX_QUERY_CHARS 的超长字符串
        huge_query = "error " * (BM25_MAX_QUERY_CHARS // 2)

        results = retriever.search(huge_query, top_k=5, extra_terms=["crash"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "id_1")
        mock_bm25.get_scores.assert_called_once()


if __name__ == "__main__":
    unittest.main()
