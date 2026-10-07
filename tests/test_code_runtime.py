"""普通对话直接读取本地与任意公开仓库，保持只读。"""
import base64
import json
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from src.chat.memory import MemoryStore
from src.chat.models import SourceIssue
from src.chat.runtime import InvestigationRuntime, DEFINITIONS
from src.chat.service import ChatService
from src.chat.store import ChatStore

class Model:
    def __init__(self, calls):
        self.calls, self.turn, self.tools = calls, 0, []
    def bind_tools(self, tools):
        self.tools = [tool["function"]["name"] for tool in tools]
        assert "edit_project_file" not in self.tools and "run_project_checks" not in self.tools
        return self
    def invoke(self, messages):
        if self.turn < len(self.calls):
            name, args = self.calls[self.turn]
            self.turn += 1
            return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{self.turn}", "type": "tool_call"}])
        results = [json.loads(m.content) for m in messages if isinstance(m, ToolMessage)]
        assert all("error" not in result for result in results)
        return AIMessage(content=json.dumps({"answer": "已读取源码", "citations": [r["evidence_id"] for r in results]}))

def run(model, store, session, text):
    memory = MemoryStore()
    memory.set_auto_capture(False)
    service = ChatService(store=store, answer_llm=model, memory_store=memory,
        retrieval_provider=lambda *_: pytest.fail("代码读取不应要求 Issue 索引"))
    message_id, _ = store.add_user_message(session.session_id, text, "one")
    service.process_turn(session.session_id, message_id)
    assert store.get(session.session_id).status == "completed"
    return store.get(session.session_id)

def test_normal_session_reads_local_paths_without_connection(tmp_path):
    roots = []
    for name in ["mine", "reference"]:
        root = tmp_path / name
        root.mkdir()
        (root / "main.py").write_text(f"value = '{name}'\n")
        roots.append(root)
    store = ChatStore(db_path=tmp_path / "sessions.db")
    session = store.create("existing-issue-repository")
    model = Model([("list_project_files", {"project_path": str(roots[0])}),
                   *[("read_project_file", {"project_path": str(root), "path": "main.py"}) for root in roots]])
    result = run(model, store, session, "读取这两个本地项目")
    assert [item.kind for item in result.evidence] == ["code", "code", "code"]
    assert result.evidence[-1].metadata["project_path"] == str(roots[1])
    assert "reference" in result.evidence[-1].text
    assert all(root.joinpath("main.py").read_text() == f"value = '{root.name}'\n" for root in roots)
    restored = ChatStore(db_path=tmp_path / "sessions.db").get(session.session_id)
    assert restored.memory_case_id and len(restored.evidence) == 3
    assert "workspace_id" not in restored.model_dump()

def test_default_local_source_needs_no_connection(tmp_path, monkeypatch):
    monkeypatch.setattr("src.chat.code_workspace.PROJECT_ROOT", tmp_path)
    (tmp_path / "main.py").write_text("default = True\n")
    store = ChatStore()
    session = store.create("existing")
    result = run(Model([("read_project_file", {"path": "main.py"})]), store, session, "读当前本地项目")
    assert "default = True" in result.evidence[-1].text

def test_current_and_other_public_repos_read_without_index(monkeypatch):
    sha = "a" * 40
    urls = []
    def response(url):
        urls.append(url)
        if "/commits/" in url:
            return {"sha": sha, "commit": {"tree": {"sha": "b" * 40}}}
        if "/git/trees/" in url:
            return {"tree": [{"type": "blob", "path": "src/main.py"}]}
        if "/contents/" in url:
            return {"type": "file", "encoding": "base64", "size": 16, "content": base64.b64encode(b"source = True\n").decode()}
        return {"private": False, "default_branch": "main"}
    monkeypatch.setattr("src.chat.code_research._request_json", response)
    store = ChatStore()
    source = SourceIssue(repository="current/project", number=1, title="当前问题", body="test", state="open", url="https://github.com/current/project/issues/1")
    session = store.create("current", source_issue=source)
    store.update(session.session_id, lambda s: setattr(s, "focus_candidate_id", "current:1"))
    model = Model([("read_repository_tree", {}), ("read_repository_file", {"path": "src/main.py", "ref": sha}),
                   ("read_repository_tree", {"repository": "https://github.com/someone/other"}),
                   ("read_repository_file", {"repository": "someone/other", "path": "src/main.py", "ref": sha})])
    result = run(model, store, session, "读当前项目，再参考别人的源码")
    assert any("/repos/current/project/contents/" in url for url in urls)
    assert any("/repos/someone/other/contents/" in url for url in urls)
    assert result.repository_id == "current" and result.focus_candidate_id == "current:1"
    assert result.evidence[-1].metadata["source_repository"] == "someone/other"

def test_removed_write_tools_cannot_execute_even_for_old_session(tmp_path):
    store = ChatStore()
    session = store.create("existing")
    runtime = InvestigationRuntime(ChatService(store=store))
    for name in ["edit_project_file", "run_project_checks"]:
        assert name not in DEFINITIONS
        with pytest.raises(ValueError, match="未知工具"):
            runtime.execute(session.session_id, name, {"path": "new.py", "new_text": "x"})
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
