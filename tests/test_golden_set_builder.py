"""golden_set_builder 的单元测试：可达性过滤、自引用清理与 family 隔离划分。"""

import unittest

from eval.golden_set_builder import build_golden_set


def _issue(issue_id: str, duplicate_of: list[str], project: str = "proj") -> dict:
    return {
        "id": issue_id,
        "title": f"title {issue_id}",
        "body": f"body {issue_id}",
        "resolution": "duplicate",
        "duplicate_of": duplicate_of,
        "project": project,
    }


class GoldenSetFilterTest(unittest.TestCase):
    def test_unreachable_targets_are_dropped(self) -> None:
        """duplicate_of 指向索引外 issue 时应剔除，全部不可达则整条丢弃。"""
        issues = [
            _issue("q1", ["t1", "missing"]),
            _issue("q2", ["missing_only"]),
        ]
        indexed = {"q1", "q2", "t1"}
        eval_set, test_set, stats = build_golden_set(issues, indexed, seed=1)

        queries = eval_set + test_set
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0]["duplicate_of"], ["t1"])
        self.assertEqual(stats["dropped_unreachable_targets"], 2)
        self.assertEqual(stats["dropped_no_reachable_target"], 1)

    def test_self_reference_is_dropped(self) -> None:
        issues = [_issue("q1", ["q1", "t1"])]
        indexed = {"q1", "t1"}
        eval_set, test_set, stats = build_golden_set(issues, indexed, seed=1)

        queries = eval_set + test_set
        self.assertEqual(queries[0]["duplicate_of"], ["t1"])
        self.assertEqual(stats["dropped_self_reference_targets"], 1)

    def test_non_duplicate_resolution_excluded(self) -> None:
        issue = _issue("q1", ["t1"])
        issue["resolution"] = "fixed"
        eval_set, test_set, _ = build_golden_set([issue], {"q1", "t1"}, seed=1)
        self.assertEqual(len(eval_set) + len(test_set), 0)


class FamilyIsolationTest(unittest.TestCase):
    def test_connected_queries_stay_on_same_side(self) -> None:
        """q1→t、q2→t 同属一个 family，任何随机种子下都不能被拆到两侧。"""
        for seed in range(20):
            issues = [
                _issue("q1", ["shared_target"]),
                _issue("q2", ["shared_target"]),
                # 一些独立 family 让划分有事可做。
                *[_issue(f"solo{i}", [f"solo_t{i}"]) for i in range(10)],
            ]
            indexed = {"shared_target", *{f"solo_t{i}" for i in range(10)}}
            indexed |= {issue["id"] for issue in issues}

            eval_set, test_set, _ = build_golden_set(issues, indexed, seed=seed)
            eval_ids = {q["id"] for q in eval_set}
            test_ids = {q["id"] for q in test_set}

            self.assertFalse(
                ("q1" in eval_ids and "q2" in test_ids)
                or ("q2" in eval_ids and "q1" in test_ids),
                f"seed={seed} 时同 family 的 q1/q2 被拆开",
            )

    def test_chain_families_merge_transitively(self) -> None:
        """a→b、b→c 通过并查集应合并为一个 family。"""
        issues = [
            _issue("a", ["b"]),
            _issue("b", ["c"]),
            *[_issue(f"solo{i}", [f"solo_t{i}"]) for i in range(8)],
        ]
        indexed = {"a", "b", "c", *{f"solo_t{i}" for i in range(8)}}
        indexed |= {issue["id"] for issue in issues}

        for seed in range(10):
            eval_set, test_set, _ = build_golden_set(issues, indexed, seed=seed)
            eval_ids = {q["id"] for q in eval_set}
            test_ids = {q["id"] for q in test_set}
            self.assertFalse(
                ("a" in eval_ids and "b" in test_ids)
                or ("b" in eval_ids and "a" in test_ids),
                f"seed={seed} 时链式 family 被拆开",
            )

    def test_split_roughly_eighty_twenty(self) -> None:
        issues = [_issue(f"q{i}", [f"t{i}"]) for i in range(100)]
        indexed = {f"t{i}" for i in range(100)} | {f"q{i}" for i in range(100)}
        eval_set, test_set, _ = build_golden_set(issues, indexed, seed=42)

        self.assertEqual(len(eval_set) + len(test_set), 100)
        self.assertGreaterEqual(len(eval_set), 70)
        self.assertLessEqual(len(eval_set), 90)


if __name__ == "__main__":
    unittest.main()
