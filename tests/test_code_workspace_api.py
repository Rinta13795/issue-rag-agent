"""本地项目入口、权限持久化和浏览器来源边界。"""
import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app
from src.chat.code_workspace import workspace_catalog
from src.chat.store import ChatStore


def test_local_session_does_not_require_issue_index(tmp_path, monkeypatch):
    monkeypatch.setenv("ISSUE_AGENT_WORKSPACES", json.dumps([str(tmp_path)]))
    workspace = workspace_catalog()[0]
    store = ChatStore()
    with patch("api.get_chat_store", return_value=store), patch("api.repository_exists", return_value=False):
        client = TestClient(app)
        assert client.get("/api/chat/workspaces").json() == [workspace]
        created = client.post("/api/chat/sessions", json={"repository_id": f"code-{workspace['id']}", "workspace_id": workspace["id"]})
        assert created.status_code == 200
        snapshot = created.json()
        assert snapshot["workspace_id"] == workspace["id"] and not snapshot["code_edits_allowed"]
        endpoint = f"/api/chat/sessions/{snapshot['session_id']}/workspace"
        enabled = client.patch(endpoint, json={"workspace_id": workspace["id"], "allow_edits": True})
        assert enabled.status_code == 200 and enabled.json()["code_edits_allowed"]
        assert client.patch(endpoint, json={"workspace_id": "unknown", "allow_edits": True}).status_code == 422
        store.update(snapshot["session_id"], lambda s: setattr(s, "status", "answering"))
        assert client.patch(endpoint, json={"workspace_id": None}).status_code == 409
        store.update(snapshot["session_id"], lambda s: setattr(s, "status", "completed"))
        disconnected = client.patch(endpoint, json={"workspace_id": None, "allow_edits": True})
        assert disconnected.status_code == 200 and not disconnected.json()["code_edits_allowed"]


def test_external_browser_origin_and_host_rejected():
    client = TestClient(app)
    assert client.get("/api/chat/workspaces", headers={"Origin": "https://outside.example"}).status_code == 403
    assert client.post("/api/chat/sessions", headers={"Origin": "null"}, json={"repository_id": "x"}).status_code == 403
    assert client.get("/api/chat/workspaces", headers={"Host": "outside.example"}).status_code == 403
    assert client.get("/api/chat/workspaces", headers={"Origin": "http://[invalid"}).status_code == 403
    assert client.get("/api/chat/workspaces", headers={"Origin": "http://localhost:5173"}).status_code == 200
