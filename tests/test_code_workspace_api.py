"""连接入口已删除，旧案例保留，普通对话保持本机访问。"""
from unittest.mock import patch
from fastapi.testclient import TestClient
from api import app
from src.chat.memory import MemoryStore
from src.chat.models import MemoryCase
from src.chat.store import ChatStore

def test_no_connection_or_permission_routes():
    client = TestClient(app)
    assert client.get("/api/chat/workspaces").status_code == 404
    assert client.patch("/api/chat/sessions/example/workspace", json={"allow_edits": True}).status_code in {404, 405}

def test_old_local_case_can_continue_without_connection():
    memory, store = MemoryStore(), ChatStore()
    memory.create_or_update_case(MemoryCase(case_id="old-local", repository_id="code-old", title="旧调查"))
    with patch("api.get_chat_store", return_value=store), patch("api.get_memory_store", return_value=memory), patch("api.repository_exists", return_value=False):
        client = TestClient(app)
        response = client.post("/api/chat/sessions", json={"repository_id": "code-old", "case_id": "old-local"})
        assert response.status_code == 200
        assert response.json()["memory_case_id"] == "old-local"
        assert "workspace_id" not in response.json() and "code_edits_allowed" not in response.json()
        assert client.post("/api/chat/sessions", json={"repository_id": "code-other", "case_id": "old-local"}).status_code == 422

def test_external_browser_origin_and_host_rejected():
    client = TestClient(app)
    for headers in [{"Origin": "https://outside.example"}, {"Origin": "null"}, {"Host": "outside.example"}, {"Origin": "http://[invalid"}]:
        assert client.get("/api/chat/sessions", headers=headers).status_code == 403
    assert client.get("/api/chat/sessions", headers={"Origin": "http://localhost:5173"}).status_code == 200
