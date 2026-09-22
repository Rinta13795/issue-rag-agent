"""测试 Component 白名单与硬过滤保护机制。"""

import unittest
from unittest.mock import MagicMock

from src.agent import nodes


class ComponentFilterProtectionTest(unittest.TestCase):
    def setUp(self):
        self._orig_retriever = nodes._retriever

    def tearDown(self):
        nodes._retriever = self._orig_retriever

    def test_component_bypasses_filter_when_not_in_whitelist(self):
        """当 LLM 提取的 component 不在白名单时，filter_dict 必须为 None，避免向量召回归零。"""
        mock_retriever = MagicMock()
        mock_retriever.search_queries.return_value = [{"id": "issue_1", "score": 0.5}]
        nodes._retriever = mock_retriever

        state = {
            "raw_issue": "crashes on pdf upload",
            "rewritten_query": "pdf upload crash",
            "keywords": ["pdf"],
            "component": "pdf_module",
        }

        result = nodes.retrieval_node(state)

        # 检查传给 search_queries 的 filter_dict
        mock_retriever.search_queries.assert_called_once()
        _, kwargs = mock_retriever.search_queries.call_args
        self.assertIsNone(kwargs["filter_dict"])
        self.assertFalse(result["component_filter_applied"])
        self.assertIn("filter bypassed", result["component_filter_note"])


if __name__ == "__main__":
    unittest.main()
