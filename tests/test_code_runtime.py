"""从工具调用到文件修改、检查和可审阅结果的离线集成测试。"""
import json

from langchain_core.messages import AIMessage, ToolMessage

from src.chat.code_workspace import workspace_catalog
from src.chat.memory import MemoryStore
from src.chat.models import ChatRepository
from src.chat.runtime import InvestigationRuntime
from src.chat.service import ChatService
from src.chat.store import ChatStore


class Model:
    def __init__(self, root):
        self.root, self.turn, self.tools = root, 0, []

    def bind_tools(self, tools):
        self.tools = [tool["function"]["name"] for tool in tools]
        return self

    def invoke(self, messages):
        self.turn += 1
        if self.turn == 1:
            name, args = "read_project_file", {"path": "main.py"}
        elif self.turn == 2:
            read = json.loads(next(m.content for m in reversed(messages) if isinstance(m, ToolMessage)))["data"]
            name, args = "edit_project_file", {"path": "main.py", "old_text": "value = 1", "new_text": "value = 2", "expected_sha256": read["sha256"]}
        elif self.turn == 3:
            name, args = "read_project_file", {"path": "main.py"}
        elif self.turn == 4:
            latest = json.loads(next(m.content for m in reversed(messages) if isinstance(m, ToolMessage)))["data"]
            assert "value = 2" in latest["content"]  # 修改后同参数读取不可使用旧缓存。
            name, args = "run_project_checks", {"check": "python-tests"}
        else:
            results = [json.loads(m.content) for m in messages if isinstance(m, ToolMessage)]
            assert results[-1]["data"]["exit_code"] == 0
            return AIMessage(content=json.dumps({"answer": "已修改并通过检查", "citations": [r["evidence_id"] for r in results]}))
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{self.turn}", "type": "tool_call"}])


def test_read_edit_reread_check_and_persisted_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("ISSUE_AGENT_WORKSPACES", json.dumps([str(tmp_path)]))
    (tmp_path / "main.py").write_text("value = 1\n")
    workspace_id = workspace_catalog()[0]["id"]
    store = ChatStore(db_path=tmp_path / "sessions.db")
    session = store.create(f"code-{workspace_id}")
    store.update(session.session_id, lambda s: (setattr(s, "workspace_id", workspace_id), setattr(s, "code_edits_allowed", True)))
    model = Model(tmp_path)
    monkeypatch.setattr("src.chat.runtime.CodeWorkspace.run_checks", lambda self, **kwargs: {"exit_code": 0, "output": "1 passed", "timed_out": False})
    service = ChatService(store=store, answer_llm=model, memory_store=MemoryStore())
    message_id, _ = store.add_user_message(session.session_id, "把 value 改为 2 并检查", "one")
    service.process_turn(session.session_id, message_id)
    result = store.get(session.session_id)
    assert result.status == "completed" and model.turn == 5
    assert (tmp_path / "main.py").read_text() == "value = 2\n"
    assert [e.kind for e in result.evidence] == ["code", "change", "code", "check"]
    assert "edit_project_file" in model.tools and "run_project_checks" in model.tools
    restored = ChatStore(db_path=tmp_path / "sessions.db").get(session.session_id)
    assert restored.workspace_id == workspace_id and restored.code_edits_allowed
    assert restored.memory_case_id
    assert restored.runtime_steps[1].result["data"]["diff"].endswith("+value = 2\n")


def test_readonly_tool_rejected_even_if_model_calls_it(tmp_path, monkeypatch):
    monkeypatch.setenv("ISSUE_AGENT_WORKSPACES", json.dumps([str(tmp_path)]))
    store = ChatStore()
    session = store.create("code-local")
    store.update(session.session_id, lambda s: setattr(s, "workspace_id", workspace_catalog()[0]["id"]))
    runtime = InvestigationRuntime(ChatService(store=store))
    import pytest
    with pytest.raises(ValueError, match="只读"):
        runtime.execute(session.session_id, "edit_project_file", {"path": "new.py", "old_text": "", "new_text": "value = 2", "expected_sha256": "missing"})
    assert not (tmp_path / "new.py").exists()


def test_external_pr_keeps_current_focus_and_source_identity(monkeypatch):
    store = ChatStore()
    session = store.create("current")
    store.add_user_message(session.session_id, "参考别的项目", "one")
    store.update(session.session_id, lambda s: setattr(s, "focus_candidate_id", "current:1"))
    service = ChatService(store=store, memory_store=MemoryStore())
    monkeypatch.setattr("src.chat.github_sync._request_json", lambda url: {"private": False})
    monkeypatch.setattr("src.chat.runtime.read_issue", lambda repo, number: {"title": "参考", "url": f"https://github.com/{repo}/issues/{number}"})
    result = InvestigationRuntime(service).execute(session.session_id, "read_issue", {"repository": "other/reference", "number": 9})
    assert result["data"]["repository"] == "other/reference"
    final = store.get(session.session_id)
    assert final.focus_candidate_id == "current:1"
    assert final.evidence[-1].metadata["source_repository"] == "other/reference"
