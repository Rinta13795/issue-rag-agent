"""decision_node 输出校验的单元测试：枚举、置信度截断、related_issues 白名单。

用假 LLM 替身注入 nodes._llm，不产生真实 API 调用。
"""

import json
import types
import unittest

from src.agent import nodes


class FakeLLM:
    """固定返回预设 JSON 文本的 LLM 替身。"""

    def __init__(self, content: str) -> None:
        self.content = content

    def invoke(self, messages: list) -> types.SimpleNamespace:
        return types.SimpleNamespace(content=self.content)


def _base_state() -> dict:
    """构造进入 decision_node 前的最小合法 State。"""
    return {
        "raw_issue": "app crashes on login",
        "rewritten_query": "login crash NullPointerException",
        "keywords": ["NullPointerException"],
        "component": None,
        "retrieved_docs": [{"id": "issue_1"}, {"id": "issue_2"}],
        "reranked_docs": [
            {"id": "issue_1", "title": "Login crash", "body": "NPE at login", "rerank_score": 0.9},
            {"id": "issue_2", "title": "Other bug", "body": "something", "rerank_score": 0.4},
        ],
        "retry_count": 0,
        "previous_decisions": [],
    }


class DecisionValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._original_llm = nodes._llm

    def tearDown(self) -> None:
        nodes._llm = self._original_llm

    def _run_with(self, payload: dict) -> dict:
        nodes._llm = FakeLLM(json.dumps(payload))
        return nodes.decision_node(_base_state())

    def test_valid_output_passes_through(self) -> None:
        result = self._run_with(
            {
                "decision": "duplicate",
                "confidence": 0.92,
                "related_issues": ["issue_1"],
                "reasoning": "same stack trace",
            }
        )
        self.assertEqual(result["decision"], "duplicate")
        self.assertAlmostEqual(result["confidence"], 0.92)
        self.assertEqual(result["related_issues"], ["issue_1"])

    def test_hallucinated_ids_are_filtered(self) -> None:
        """候选外的 issue ID 必须被白名单过滤（v1 会原样返回幻觉 ID）。"""
        result = self._run_with(
            {
                "decision": "duplicate",
                "confidence": 0.9,
                "related_issues": ["issue_999", "issue_1"],
                "reasoning": "…",
            }
        )
        self.assertEqual(result["related_issues"], ["issue_1"])

    def test_duplicate_without_valid_ids_downgrades_to_new(self) -> None:
        """全部 related ID 都是幻觉时，duplicate 判断失去证据，应降级为 new。"""
        result = self._run_with(
            {
                "decision": "duplicate",
                "confidence": 0.9,
                "related_issues": ["ghost_1", "ghost_2"],
                "reasoning": "…",
            }
        )
        self.assertEqual(result["decision"], "new")
        self.assertEqual(result["related_issues"], [])
        # 降级后置信度不超过 0.4，保证在次数允许时触发 retry。
        self.assertLessEqual(result["confidence"], 0.4)

    def test_invalid_enum_becomes_low_confidence_new(self) -> None:
        result = self._run_with(
            {
                "decision": "duplicated!!",
                "confidence": 0.95,
                "related_issues": ["issue_1"],
                "reasoning": "…",
            }
        )
        self.assertEqual(result["decision"], "new")
        self.assertEqual(result["confidence"], 0.0)

    def test_confidence_clamped_to_unit_interval(self) -> None:
        result = self._run_with(
            {
                "decision": "similar",
                "confidence": 1.7,
                "related_issues": ["issue_2"],
                "reasoning": "…",
            }
        )
        self.assertEqual(result["confidence"], 1.0)

        result = self._run_with(
            {
                "decision": "new",
                "confidence": -0.3,
                "related_issues": [],
                "reasoning": "…",
            }
        )
        self.assertEqual(result["confidence"], 0.0)

    def test_history_records_top_candidates_for_retry(self) -> None:
        """previous_decisions 应保存 Top 候选的 id/title，供重试 prompt 展示。"""
        result = self._run_with(
            {
                "decision": "new",
                "confidence": 0.2,
                "related_issues": [],
                "reasoning": "…",
            }
        )
        last = result["previous_decisions"][-1]
        self.assertIn("top_candidates", last)
        self.assertEqual(last["top_candidates"][0]["id"], "issue_1")
        self.assertEqual(last["top_candidates"][0]["title"], "Login crash")

    def test_unparseable_output_falls_back_to_new(self) -> None:
        nodes._llm = FakeLLM("I think this is probably a duplicate of something.")
        result = nodes.decision_node(_base_state())
        self.assertEqual(result["decision"], "new")
        self.assertEqual(result["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
