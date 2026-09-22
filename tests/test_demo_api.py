"""测试 Demo FastAPI 接口：/health, /api/examples, /api/system, /api/evaluation/summary, /api/runs。"""

import unittest
from fastapi.testclient import TestClient

from api import app


class DemoApiTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"status": "ok"})

    def test_examples(self):
        res = self.client.get("/api/examples")
        self.assertEqual(res.status_code, 200)
        examples = res.json()
        self.assertGreaterEqual(len(examples), 3)
        self.assertIn("title", examples[0])
        self.assertIn("expected_decision", examples[0])

    def test_system_info(self):
        res = self.client.get("/api/system")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["version"], "2.0.0")
        self.assertTrue(data["bm25_ready"])
        self.assertTrue(data["vector_ready"])
        # 不得泄漏敏感字段
        self.assertNotIn("api_key", data)
        self.assertNotIn("secret", data)

    def test_evaluation_summary(self):
        res = self.client.get("/api/evaluation/summary")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("metrics", data)
        self.assertIn("notes", data)
        # 确保包含 Recall 不是 accuracy 的说明
        notes_str = " ".join(data["notes"])
        self.assertIn("非分类准确率", notes_str)

    def test_create_and_get_run(self):
        from unittest.mock import patch
        with patch("src.demo.runner.ObservableRunner.submit_run") as mock_submit:
            res = self.client.post("/api/runs", json={"issue_text": "testing api run"})
            self.assertEqual(res.status_code, 200)
            run_data = res.json()
            self.assertIn("run_id", run_data)
            self.assertEqual(run_data["status"], "queued")
            mock_submit.assert_called_once()

        run_id = run_data["run_id"]
        res2 = self.client.get(f"/api/runs/{run_id}")
        self.assertEqual(res2.status_code, 200)
        snapshot = res2.json()
        self.assertEqual(snapshot["run_id"], run_id)

    def test_get_non_existent_run(self):
        res = self.client.get("/api/runs/non_existent_id")
        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
