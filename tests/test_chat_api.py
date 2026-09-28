"""会话 API 的本地契约，不发起真实 LLM 调用。"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app
from src.chat.store import ChatStore


class FakeRunner:
    def __init__(self):
        self.submitted = []

    def submit(self, session_id, message_id):
        self.submitted.append((session_id, message_id))


def test_session_create_send_get_and_idempotency():
    store = ChatStore()
    runner = FakeRunner()
    with patch("api.get_chat_store", return_value=store), patch("api.get_chat_runner", return_value=runner), patch("api.repository_exists", return_value=True):
        client = TestClient(app)
        created = client.post("/api/chat/sessions", json={"repository_id": "vscode"})
        assert created.status_code == 200
        session_id = created.json()["session_id"]
        assert created.json()["repository_id"] == "vscode"
        assert client.get("/api/chat/sessions").json()[0]["repository_id"] == "vscode"
        payload = {"content": "登录失败", "client_message_id": "local-1"}
        first = client.post(f"/api/chat/sessions/{session_id}/messages", json=payload)
        second = client.post(f"/api/chat/sessions/{session_id}/messages", json=payload)
        assert first.status_code == second.status_code == 202
        assert first.json()["message_id"] == second.json()["message_id"]
        assert len(runner.submitted) == 1
        snapshot = client.get(f"/api/chat/sessions/{session_id}").json()
        assert snapshot["messages"][0]["content"] == "登录失败"
        assert snapshot["status"] == "thinking"
        assert "api_key" not in snapshot
        assert client.post(f"/api/chat/sessions/{session_id}/messages", json={
            "content": "别的消息", "client_message_id": "local-1",
        }).status_code == 409
        assert client.post(f"/api/chat/sessions/{session_id}/messages", json={
            "content": "补充 401", "client_message_id": "local-2",
        }).status_code == 409


def test_invalid_or_expired_session_and_empty_message():
    store = ChatStore(ttl_seconds=0)
    with patch("api.get_chat_store", return_value=store), patch("api.get_chat_runner", return_value=FakeRunner()), patch("api.repository_exists", return_value=True):
        client = TestClient(app)
        assert client.get("/api/chat/sessions/unknown").status_code == 404
        assert client.post("/api/chat/sessions/unknown/messages", json={
            "content": "hi", "client_message_id": "local-1",
        }).status_code == 404
        session_id = client.post("/api/chat/sessions", json={"repository_id": "vscode"}).json()["session_id"]
        assert client.get(f"/api/chat/sessions/{session_id}").status_code == 404
    with patch("api.get_chat_store", return_value=ChatStore()), patch("api.repository_exists", return_value=True):
        client = TestClient(app)
        session_id = client.post("/api/chat/sessions", json={"repository_id": "vscode"}).json()["session_id"]
        assert client.post(f"/api/chat/sessions/{session_id}/messages", json={
            "content": "   ", "client_message_id": "local-1",
        }).status_code == 422


def test_new_session_requires_an_indexed_repository():
    with patch("api.repository_exists", return_value=False):
        client = TestClient(app)
        assert client.post("/api/chat/sessions", json={"repository_id": "unknown"}).status_code == 422
        assert client.post("/api/chat/sessions", json={}).status_code == 422
