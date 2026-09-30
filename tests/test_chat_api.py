"""会话 API 的本地契约，不发起真实 LLM 调用。"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app
from src.chat.store import ChatStore
from src.chat.models import ChatRepository, MemoryProposal, SourceIssue
from src.chat.memory import MemoryStore


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


def test_github_sync_api_validation_and_status():
    class FakeManager:
        def start(self, value):
            if value == "invalid":
                raise ValueError("仓库格式错误")
            from src.chat.models import SyncRepositoryStatus
            return SyncRepositoryStatus(job_id="sync_1", repository=value, status="fetching", message="正在读取公开 Issue")

        def get(self, job_id):
            return self.start("owner/repo") if job_id == "sync_1" else None

    with patch("api.get_sync_manager", return_value=FakeManager()):
        client = TestClient(app)
        assert client.post("/api/chat/repositories/sync", json={"repository": "invalid"}).status_code == 422
        response = client.post("/api/chat/repositories/sync", json={"repository": "owner/repo"})
        assert response.status_code == 202
        assert response.json()["status"] == "fetching"
        assert client.get("/api/chat/repositories/sync/sync_1").status_code == 200
        assert client.get("/api/chat/repositories/sync/missing").status_code == 404


def test_retry_endpoint_resubmits_same_message_without_duplicate():
    store = ChatStore()
    runner = FakeRunner()
    session_id = store.create("vscode").session_id
    message_id, _ = store.add_user_message(session_id, "构建失败", "local-1")
    store.update(session_id, lambda session: setattr(session, "status", "failed"))
    with patch("api.get_chat_store", return_value=store), patch("api.get_chat_runner", return_value=runner):
        client = TestClient(app)
        response = client.post(f"/api/chat/sessions/{session_id}/retry")
        assert response.status_code == 202
        assert response.json()["message_id"] == message_id
        assert runner.submitted == [(session_id, message_id)]
        assert len(store.get(session_id).messages) == 1
        assert client.post(f"/api/chat/sessions/{session_id}/retry").status_code == 409
        assert client.post("/api/chat/sessions/missing/retry").status_code == 404


def test_import_issue_binds_authoritative_repository():
    source = SourceIssue(repository="Owner/Repo", number=12, title="Crash", body="Steps", state="open",
                         url="https://github.com/Owner/Repo/issues/12")
    store = ChatStore()
    from src.chat.github_sync import repository_id
    with patch("api.get_chat_store", return_value=store), patch("api.repository_exists", return_value=True), patch("api.fetch_issue", return_value=source):
        client = TestClient(app)
        preview = client.post("/api/chat/issues/preview", json={"issue_url": source.url})
        assert preview.status_code == 200
        response = client.post("/api/chat/sessions", json={"repository_id": repository_id("Owner/Repo"), "issue_url": source.url})
        assert response.status_code == 200
        assert response.json()["source_issue"]["number"] == 12
        assert client.get("/api/chat/sessions").json()[0]["title"] == "Crash"
        wrong = client.post("/api/chat/sessions", json={"repository_id": "other", "issue_url": source.url})
        assert wrong.status_code == 422


def test_issue_draft_publish_requires_confirmation_and_is_idempotent():
    store = ChatStore()
    session_id = store.create("gh-repo").session_id
    draft = store.save_issue_draft(session_id, "App fails to start", "Steps to reproduce").issue_draft
    repo = ChatRepository(id="gh-repo", label="Owner/Repo", issue_count=10, source="GitHub", github_url="https://github.com/Owner/Repo")
    calls = []

    def fake_publish(repository, title, body):
        calls.append((repository, title, body))
        return "https://github.com/Owner/Repo/issues/99", 99

    with patch("api.get_chat_store", return_value=store), patch("api.list_repositories", return_value=[repo]), patch("api.create_github_issue", side_effect=fake_publish):
        client = TestClient(app)
        url = f"/api/chat/sessions/{session_id}/issue-draft/publish"
        payload = {"draft_id": draft.draft_id, "version": draft.version, "confirmed": False}
        assert client.post(url, json=payload).status_code == 422
        assert not calls
        edit = client.patch(f"/api/chat/sessions/{session_id}/issue-draft", json={
            "title": "Updated title", "body": "New steps", "version": draft.version,
        })
        assert edit.status_code == 200
        payload.update({"version": draft.version + 1, "confirmed": True})
        first = client.post(url, json=payload)
        second = client.post(url, json=payload)
        assert first.status_code == second.status_code == 200
        assert len(calls) == 1
        assert calls[0] == ("Owner/Repo", "Updated title", "New steps")
        assert second.json()["issue_draft"]["published_number"] == 99


def test_memory_requires_proposal_and_can_be_deleted():
    store = ChatStore()
    memory = MemoryStore()
    session_id = store.create("repo-a").session_id
    proposal = MemoryProposal(kind="preference", scope="global", text="先查重", source_excerpt="先查重")
    store.update(session_id, lambda session: setattr(session, "memory_proposal", proposal))
    with patch("api.get_chat_store", return_value=store), patch("api.get_memory_store", return_value=memory):
        client = TestClient(app)
        url = f"/api/chat/sessions/{session_id}/memory"
        assert client.post(url, json={"kind": "preference", "scope": "global", "text": "先查重", "confirmed": False}).status_code == 422
        saved = client.post(url, json={"kind": "preference", "scope": "global", "text": "先查重", "confirmed": True})
        assert saved.status_code == 200
        assert saved.json()["repository_id"] is None
        assert client.post(url, json={"kind": "preference", "scope": "global", "text": "先查重", "confirmed": True}).status_code == 409
        memory_id = saved.json()["memory_id"]
        assert len(client.get("/api/chat/memories?repository_id=repo-a").json()) == 1
        assert client.delete(f"/api/chat/memories/{memory_id}").status_code == 200
        assert client.get("/api/chat/memories?repository_id=repo-a").json() == []


def test_uncertain_publish_is_not_retried():
    from src.chat.github_sync import GitHubPublishError

    store = ChatStore()
    session_id = store.create("gh-repo").session_id
    draft = store.save_issue_draft(session_id, "Title", "Body").issue_draft
    repo = ChatRepository(id="gh-repo", label="Owner/Repo", issue_count=10, source="GitHub", github_url="https://github.com/Owner/Repo")
    with patch("api.get_chat_store", return_value=store), patch("api.list_repositories", return_value=[repo]), patch(
        "api.create_github_issue", side_effect=GitHubPublishError("网络结果不确定", definite=False),
    ) as publisher:
        client = TestClient(app)
        url = f"/api/chat/sessions/{session_id}/issue-draft/publish"
        payload = {"draft_id": draft.draft_id, "version": 1, "confirmed": True}
        assert client.post(url, json=payload).status_code == 502
        assert store.get(session_id).issue_draft.status == "uncertain"
        assert client.post(url, json=payload).status_code == 409
        assert publisher.call_count == 1


def test_answer_endpoint_binds_question_once_and_stale_answer_is_rejected():
    from src.chat.models import PendingQuestion, RuntimeStep
    store, runner = ChatStore(), FakeRunner()
    session = store.create("vscode")
    store.update(session.session_id, lambda s: (
        setattr(s, "pending_question", PendingQuestion(question_id="q1",call_id="call",question="版本？",turn_id="m1")),
        setattr(s, "status", "waiting_for_user"),
        s.runtime_steps.append(RuntimeStep(call_id="call",turn_id="m1",tool="ask_user",status="waiting")),
    ))
    with patch("api.get_chat_store", return_value=store), patch("api.get_chat_runner", return_value=runner):
        client = TestClient(app)
        url = f"/api/chat/sessions/{session.session_id}/answers"
        data = {"question_id":"q1","client_message_id":"answer-1","answer":"v2"}
        assert client.post(url,json=data).status_code == 202
        assert client.post(url,json=data).status_code == 202
        assert len(runner.submitted) == 1
        assert client.post(url,json={**data,"client_message_id":"answer-2"}).status_code == 409
        assert store.get(session.session_id).runtime_resume
