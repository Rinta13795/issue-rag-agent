"""eval/metrics.py 的单元测试：重点验证 v2 修复后的 nDCG IDCG 计算。"""

import math
import unittest

from eval.metrics import compute_all, mrr, ndcg_at_k, precision_at_k, recall_at_k


class RecallPrecisionMrrTest(unittest.TestCase):
    def test_recall_counts_hits_over_golden_size(self) -> None:
        retrieved = ["a", "b", "c", "d"]
        golden = ["b", "z"]
        # Top-4 命中 b，golden 共 2 个目标 -> 0.5。
        self.assertAlmostEqual(recall_at_k(retrieved, golden, 4), 0.5)

    def test_precision_uses_k_as_denominator(self) -> None:
        retrieved = ["a", "b", "c", "d", "e"]
        golden = ["a", "e"]
        self.assertAlmostEqual(precision_at_k(retrieved, golden, 5), 2 / 5)

    def test_mrr_returns_reciprocal_of_first_hit(self) -> None:
        self.assertAlmostEqual(mrr(["x", "y", "g"], ["g"]), 1 / 3)
        self.assertAlmostEqual(mrr(["x", "y"], ["g"]), 0.0)


class NdcgTest(unittest.TestCase):
    def test_perfect_ranking_scores_one(self) -> None:
        self.assertAlmostEqual(ndcg_at_k(["g1", "g2"], ["g1", "g2"], 10), 1.0)

    def test_idcg_uses_full_golden_size_not_retrieved_gains(self) -> None:
        """v1 bug：IDCG 从已检索 gains 计算，命中 1 条就得满分。

        golden 有 2 个目标、只命中排名第 1 的 1 条时：
        DCG = 1/log2(2) = 1.0；IDCG 必须按 2 个理想命中算 = 1/log2(2) + 1/log2(3)。
        v1 错误实现会返回 1.0，正确值应显著小于 1。
        """
        golden = ["g1", "g2"]
        retrieved = ["g1", "x", "y"]
        expected_idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        expected = 1.0 / expected_idcg
        self.assertAlmostEqual(ndcg_at_k(retrieved, golden, 10), expected, places=6)
        self.assertLess(ndcg_at_k(retrieved, golden, 10), 1.0)

    def test_idcg_capped_by_k(self) -> None:
        # golden 有 20 个目标但 k=10 时，理想命中最多 10 个，IDCG 以 10 封顶。
        golden = [f"g{i}" for i in range(20)]
        retrieved = golden[:10]
        self.assertAlmostEqual(ndcg_at_k(retrieved, golden, 10), 1.0)

    def test_no_hit_returns_zero(self) -> None:
        self.assertAlmostEqual(ndcg_at_k(["x", "y"], ["g"], 10), 0.0)


class ComputeAllTest(unittest.TestCase):
    def test_default_cutoffs(self) -> None:
        result = compute_all(["a"], ["a"])
        self.assertIn("recall@5", result)
        self.assertIn("recall@30", result)

    def test_custom_cutoffs_for_rerank(self) -> None:
        """Rerank 输出最多 10 条，只应在 @5/@10 上计算，不应出现 recall@30。"""
        result = compute_all(["a"], ["a"], cutoffs=(5, 10))
        self.assertIn("recall@5", result)
        self.assertIn("recall@10", result)
        self.assertNotIn("recall@30", result)


if __name__ == "__main__":
    unittest.main()
