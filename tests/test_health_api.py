"""存活探针 /health：不触发模型加载，聊天前端靠它判断后端是否可用。"""

import unittest
from fastapi.testclient import TestClient

from api import app


class HealthApiTest(unittest.TestCase):
    def test_health(self):
        res = TestClient(app).get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
