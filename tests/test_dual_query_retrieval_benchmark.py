"""双 Query Benchmark 指标计算的快速单元测试。"""

import math
import unittest

from eval.benchmarks.dual_query_retrieval import _ranking_metrics, summarize


class RankingMetricsTest(unittest.TestCase):
    def test_ndcg_uses_all_expected_ids_for_ideal_ranking(self) -> None:
        metrics = _ranking_metrics(
            retrieved_ids=["duplicate-a"],
            expected_ids=["duplicate-a", "duplicate-b"],
            cutoffs=(1, 2),
        )

        self.assertEqual(metrics["recall@1"], 0.5)
        self.assertEqual(metrics["recall@2"], 0.5)
        self.assertTrue(
            math.isclose(metrics["ndcg@10"], 0.6131, abs_tol=0.0001)
        )

    def test_mrr_uses_first_relevant_rank(self) -> None:
        metrics = _ranking_metrics(
            retrieved_ids=["wrong", "duplicate-b", "duplicate-a"],
            expected_ids=["duplicate-a", "duplicate-b"],
            cutoffs=(3,),
        )

        self.assertEqual(metrics["hit_rate@3"], 1.0)
        self.assertEqual(metrics["recall@3"], 1.0)
        self.assertEqual(metrics["mrr"], 0.5)

    def test_summary_reports_rescue_and_case_type(self) -> None:
        empty_metrics = {
            "hit_rate@5": 0.0,
            "recall@5": 0.0,
            "hit_rate@10": 0.0,
            "recall@10": 0.0,
            "hit_rate@30": 0.0,
            "recall@30": 0.0,
            "mrr": 0.0,
            "ndcg@10": 0.0,
        }
        hit_metrics = {key: 1.0 for key in empty_metrics}
        records = [
            {
                "case_id": "case-1",
                "case_type": "rewrite_drift",
                "single_rewritten": {
                    "retrieval_metrics": empty_metrics,
                    "rerank_metrics": empty_metrics,
                    "retrieval_latency_seconds": 1.0,
                    "rerank_latency_seconds": 1.0,
                    "total_latency_seconds": 2.0,
                },
                "dual_query": {
                    "retrieval_metrics": hit_metrics,
                    "rerank_metrics": hit_metrics,
                    "retrieval_latency_seconds": 2.0,
                    "rerank_latency_seconds": 1.0,
                    "total_latency_seconds": 3.0,
                },
            }
        ]

        summary = summarize(records, top_k=30)

        self.assertEqual(
            summary["safety_checks"]["retrieval_rescue_count@30"],
            1,
        )
        self.assertEqual(summary["safety_checks"]["rerank_rescue_count@10"], 1)
        self.assertIn("rewrite_drift", summary["by_case_type"])


if __name__ == "__main__":
    unittest.main()
