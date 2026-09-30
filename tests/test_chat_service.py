"""草稿与手动经验提炼保持用户确认入口；Runtime 回归见 test_chat_runtime。"""

import json
from types import SimpleNamespace

from src.chat.models import ChatCandidate
from src.chat.service import ChatService
from src.chat.store import ChatStore


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return SimpleNamespace(content=json.dumps(self.responses.pop(0), ensure_ascii=False), usage_metadata={"input_tokens": 10, "output_tokens": 5})


class FakeRetriever:
    def __init__(self, docs=None):
        self.docs = docs if docs is not None else [{"id": "vscode:101", "title": "Login returns 401", "body": "After clicking login, API returns 401."}]
        self.queries = []

    def search(self, query, top_k, project=None):
        self.queries.append(query)
        return self.docs


class FakeReranker:
    def rerank(self, query, docs):
        return [dict(doc, rerank_score=0.9) for doc in docs]


def send(service, store, session_id, content, client_id):
    message_id, _ = store.add_user_message(session_id, content, client_id)
    service.process_turn(session_id, message_id)
    return store.get(session_id)


def test_draft_and_memory_proposals_do_not_publish_or_save_automatically(monkeypatch):
    from src.chat.models import SourceIssue
    from src.chat.memory import MemoryStore

    store = ChatStore()
    memory = MemoryStore()
    source = SourceIssue(repository="Owner/Repo", number=12, title="Crash", body="Crashes after clicking start",
                         state="open", url="https://github.com/Owner/Repo/issues/12")
    session = store.create("gh-repo", source_issue=source)
    store.add_user_message(session.session_id, "我确认点击开始后会崩溃", "one")
    store.update(session.session_id, lambda item: setattr(item, "status", "completed"))
    answer = FakeLLM([
        {"title": "App crashes on start", "body": "实际行为：点击开始后崩溃。\n复现步骤：待补充。"},
        {"kind": "experience", "scope": "repository", "text": "点击开始后崩溃，等待进一步定位", "source_excerpt": "我确认点击开始后会崩溃"},
    ])
    monkeypatch.setattr("src.chat.service.list_repositories", lambda: [])
    service = ChatService(store, FakeLLM([]), answer, memory_store=memory)
    drafted = service.create_draft(session.session_id)
    assert drafted.issue_draft.status == "draft"
    proposed = service.create_memory_proposal(session.session_id)
    assert proposed.memory_proposal.kind == "experience"
    assert memory.list() == []
