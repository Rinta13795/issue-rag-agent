"""Agent 任务评测的标注与评分逻辑单测；不访问 GitHub，不调用模型。"""

import unittest

from eval.benchmarks.agent_tasks import closes_issue, label_case, score_record, summarize


def _case(resolution: str, fix_prs: list[int], case_id: str = "a/b#1") -> dict:
    return {"id": case_id, "expected": {"resolution": resolution, "fix_prs": fix_prs}}


class LabelCaseTest(unittest.TestCase):
    def test_closing_keyword_must_point_to_this_issue(self) -> None:
        self.assertTrue(closes_issue("This PR fixes #12.", 12))
        self.assertTrue(closes_issue("Closes owner/repo#12", 12))
        self.assertFalse(closes_issue("Related to #12", 12))
        self.assertFalse(closes_issue("fixes #123", 12))

    def test_closed_issue_with_merged_closing_pr_is_fixed(self) -> None:
        issue = {"state": "closed", "linked_prs": [{"number": 5}]}
        prs = [{"number": 5, "merged": True, "body": "Fixes #12"}]
        self.assertEqual(label_case(issue, prs, 12), {"resolution": "fixed_merged", "fix_prs": [5]})

    def test_unmerged_or_unrelated_pr_is_left_for_humans(self) -> None:
        issue = {"state": "closed", "linked_prs": [{"number": 5}]}
        self.assertIsNone(label_case(issue, [{"number": 5, "merged": False, "body": "Fixes #12"}], 12))
        self.assertIsNone(label_case(issue, [{"number": 5, "merged": True, "body": "Refactor"}], 12))

    def test_open_issue_without_links_is_no_fix_only_if_timeline_was_read(self) -> None:
        self.assertEqual(label_case({"state": "open", "linked_prs": []}, [], 12),
                         {"resolution": "no_fix", "fix_prs": []})
        self.assertIsNone(label_case({"state": "open", "linked_prs": [], "links_error": "rate limited"}, [], 12))


class ScoreRecordTest(unittest.TestCase):
    def test_fix_case_needs_reading_and_citing_the_fix_pr(self) -> None:
        record = {
            "status": "completed",
            "steps": [{"tool": "read_issue", "arguments": {"number": 1}, "status": "completed"},
                      {"tool": "read_pr", "arguments": {"number": 5}, "status": "completed"}],
            "citations": ["pr:a/b#5"], "answer": "已由 PR 修复", "model_calls": 3,
            "prompt_tokens": 100, "completion_tokens": 20, "latency_seconds": 2.0,
        }
        score = score_record(_case("fixed_merged", [5]), record)
        self.assertTrue(score["key_evidence_hit"])
        self.assertTrue(score["citation_hit"])
        self.assertIsNone(score["overclaim"])
        self.assertEqual(score["total_tokens"], 120)

    def test_reading_a_different_pr_is_not_key_evidence(self) -> None:
        record = {"status": "completed", "citations": ["pr:a/b#9"],
                  "steps": [{"tool": "read_pr", "arguments": {"number": 9}, "status": "completed"}]}
        score = score_record(_case("fixed_merged", [5]), record)
        self.assertFalse(score["key_evidence_hit"])
        self.assertFalse(score["citation_hit"])

    def test_no_fix_case_flags_overclaim(self) -> None:
        claimed = score_record(_case("no_fix", []), {"status": "completed", "answer": "这个问题已修复"})
        honest = score_record(_case("no_fix", []), {"status": "completed", "answer": "目前没有找到修复 PR"})
        self.assertTrue(claimed["overclaim"])
        self.assertFalse(honest["overclaim"])
        self.assertIsNone(claimed["key_evidence_hit"])


class SummarizeTest(unittest.TestCase):
    def test_rates_skip_inapplicable_cases_and_report_consistency(self) -> None:
        fix_hit = {"status": "completed", "citations": ["pr:a/b#5"],
                   "steps": [{"tool": "read_pr", "arguments": {"number": 5}, "status": "completed"}]}
        fix_miss = {"status": "completed", "citations": [], "steps": []}
        scores = [
            score_record(_case("fixed_merged", [5]), {**fix_hit, "repeat": 1}),
            score_record(_case("fixed_merged", [5]), {**fix_miss, "repeat": 2}),
            score_record(_case("no_fix", [], "a/b#2"), {"status": "completed", "answer": "没有修复", "repeat": 1}),
        ]
        summary = summarize(scores)
        self.assertEqual(summary["key_evidence_recall"], 0.5)
        self.assertEqual(summary["overclaim_rate"], 0.0)
        self.assertEqual(summary["consistency_rate"], 0.0)
        self.assertIn("fixed_merged", summary["by_resolution"])


if __name__ == "__main__":
    unittest.main()
