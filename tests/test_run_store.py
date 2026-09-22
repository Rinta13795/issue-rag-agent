"""测试 RunStore 状态转换、容量控制与 TTL 清理。"""

import unittest
from src.demo.models import NodeStatus, RunStatus
from src.demo.run_store import RunStore


class RunStoreTest(unittest.TestCase):
    def test_create_and_lifecycle(self):
        store = RunStore(max_runs=5, ttl_seconds=60)
        snapshot = store.create_run("run_1", "test issue")
        self.assertEqual(snapshot.run_id, "run_1")
        self.assertEqual(snapshot.status, RunStatus.QUEUED)
        self.assertEqual(len(snapshot.nodes), 4)

        # 启动节点
        store.mark_node_started("run_1", "query_analysis")
        s2 = store.get_run("run_1")
        self.assertEqual(s2.status, RunStatus.RUNNING)
        self.assertEqual(s2.current_node, "query_analysis")
        self.assertEqual(s2.nodes[0].status, NodeStatus.RUNNING)

        # 完成节点
        store.mark_node_completed("run_1", "query_analysis", 120, summary="done")
        s3 = store.get_run("run_1")
        self.assertEqual(s3.nodes[0].status, NodeStatus.COMPLETED)
        self.assertEqual(s3.nodes[0].elapsed_ms, 120)

        # 完成 run
        store.mark_completed("run_1", 500)
        s4 = store.get_run("run_1")
        self.assertEqual(s4.status, RunStatus.COMPLETED)
        self.assertEqual(s4.total_elapsed_ms, 500)

    def test_max_runs_capacity(self):
        store = RunStore(max_runs=3, ttl_seconds=60)
        for i in range(5):
            store.create_run(f"run_{i}", f"issue {i}")
        self.assertIsNone(store.get_run("run_0"))
        self.assertIsNone(store.get_run("run_1"))
        self.assertIsNotNone(store.get_run("run_2"))
        self.assertIsNotNone(store.get_run("run_3"))
        self.assertIsNotNone(store.get_run("run_4"))


if __name__ == "__main__":
    unittest.main()
