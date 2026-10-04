"""原生工具调用协议的离线回归，禁止测试访问 GitHub。"""
import json
import pytest
from langchain_core.messages import AIMessage, ToolMessage, messages_from_dict
from src.chat.models import SourceIssue
from src.chat.memory import MemoryStore
from src.chat.service import ChatService
from src.chat.store import ChatStore
from src.chat.runtime import InvestigationRuntime

class Model:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
    def bind_tools(self, tools):
        assert len(tools) == 6
        return self
    def invoke(self, messages):
        self.requests.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception): raise response
        return response

def tool(name, args, call_id="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])
def reply(text="调查完成", citations=None):
    return AIMessage(content=json.dumps({"answer": text, "citations": citations or []}, ensure_ascii=False))
@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr("src.chat.runtime.list_repositories", lambda: [])
    monkeypatch.setattr("src.chat.runtime.search_live_issues", lambda *a, **k: [])

def make(model, store=None, memory=None, issue=True):
    store = store or ChatStore()
    memory = memory or MemoryStore()
    memory.set_auto_capture(False)
    source = SourceIssue(repository="Owner/Repo", number=352, title="INFO 日志", body="send_tool_hints=False", state="open", url="https://github.com/Owner/Repo/issues/352") if issue else None
    session = store.create("repo", source_issue=source)
    service = ChatService(store=store, answer_llm=model, memory_store=memory)
    return store, service, session

def send(store, service, session, text="维护者怎么解决？", key="one"):
    message_id, _ = store.add_user_message(session.session_id, text, key)
    service.process_turn(session.session_id, message_id)
    return store.get(session.session_id)

def test_followup_reads_linked_pr_and_rejects_fake_citation(monkeypatch):
    model = Model(tool("read_issue", {"number":352}), tool("read_pr", {"number":359}, "c2"), reply("PR 已合并，是否发布仍未知", ["pr:Owner/Repo#359", "fabricated"]))
    store, service, session = make(model)
    seen = []
    def issue(repo, number):
        seen.append(("issue", number))
        return {"title":"INFO", "url":session.source_issue.url, "linked_prs":[{"number":359, "relationship":"cross_referenced"}]}
    def pr(repo, number):
        seen.append(("pr", number))
        return {"title":"fix INFO", "url":"https://github.com/Owner/Repo/pull/359", "merged":True, "body":"Fixes #352"}
    monkeypatch.setattr("src.chat.runtime.read_issue", issue)
    monkeypatch.setattr("src.chat.runtime.read_pr", pr)
    result = send(store, service, session)
    assert result.status == "completed"
    assert seen == [("issue",352),("pr",359)]
    assert result.messages[-1].citations == ["pr:Owner/Repo#359"]
    assert result.model_calls == 3 and len(result.evidence) == 2
    assert len([m for m in model.requests[-1] if isinstance(m, ToolMessage)]) == 2

def test_question_restart_idempotence_and_resume(tmp_path):
    db = tmp_path / "sessions.db"
    model = Model(tool("ask_user", {"question":"哪个版本？", "options":["v1","v2"]}))
    store, service, session = make(model, ChatStore(db_path=db))
    paused = send(store, service, session)
    assert paused.status == "waiting_for_user"
    restored = ChatStore(db_path=db)
    pending = restored.get(session.session_id).pending_question
    assert pending.question == "哪个版本？"
    message_id, created = restored.answer_question(session.session_id, pending.question_id, "v2", "answer")
    assert created
    assert restored.answer_question(session.session_id, pending.question_id, "v2", "answer") == (message_id, False)
    with pytest.raises(RuntimeError): restored.answer_question(session.session_id, pending.question_id, "v1", "other")
    resumed_model = Model(reply("按 v2 继续"))
    resumed_service = ChatService(store=restored, answer_llm=resumed_model, memory_store=service.memory_store)
    resumed_service.process_turn(session.session_id, message_id)
    result = restored.get(session.session_id)
    assert result.status == "completed" and result.pending_question is None
    answers = [json.loads(m.content) for m in resumed_model.requests[0] if isinstance(m, ToolMessage)]
    assert answers[-1]["answer"] == "v2"
    assert result.messages[-2].role == "user" and result.messages[-2].content == "v2"

def test_plain_chat_answers_pending_and_cancel_releases_it():
    model = Model(tool("ask_user", {"question":"版本？"}), reply("继续"), tool("ask_user", {"question":"环境？"}, "c2"))
    store, service, session = make(model)
    send(store, service, session)
    answered = send(store, service, session, "v3", "two")
    assert answered.status == "completed"
    paused = send(store, service, session, "再查", "three")
    q = paused.pending_question
    mid, _ = store.answer_question(session.session_id, q.question_id, "", "cancel", True)
    service.process_turn(session.session_id, mid)
    assert store.get(session.session_id).status == "completed" and len(model.requests) == 3

def test_repeated_tools_cached_errors_return_to_model(monkeypatch):
    model = Model(tool("read_pr", {"number":3}), tool("read_pr", {"number":3}, "c2"), tool("read_issue", {"number":-1}, "c3"), reply("资料不足"))
    store, service, session = make(model)
    calls = []
    monkeypatch.setattr("src.chat.runtime.read_pr", lambda *a: calls.append(a) or {"title":"fix", "url":"https://github.com/Owner/Repo/pull/3"})
    result = send(store, service, session)
    assert len(calls) == 1 and result.status == "completed"
    assert result.runtime_steps[-1].status == "failed"
    assert "error" in json.loads([m for m in model.requests[-1] if isinstance(m, ToolMessage)][-1].content)

def test_tool_and_model_budgets(monkeypatch):
    batch = lambda n: AIMessage(content="", tool_calls=[{"name":"read_pr","args":{"number":k+1},"id":f"c{n}-{k}","type":"tool_call"} for k in range(4)])
    model = Model(batch(1), batch(2), batch(3), reply("预算到，仍有未知"))
    store, service, session = make(model)
    calls = []
    monkeypatch.setattr("src.chat.runtime.read_pr", lambda *a: calls.append(a) or {"title":"fix", "url":"https://github.com/Owner/Repo/pull/3"})
    result = send(store, service, session)
    assert result.model_calls == 4 and len(calls) == 4
    assert len([s for s in result.runtime_steps if s.status == "failed"]) == 2

def test_ask_in_batch_preserves_all_tool_results(monkeypatch):
    response = AIMessage(content="", tool_calls=[{"name":"ask_user","args":{"question":"版本？"},"id":"ask","type":"tool_call"},{"name":"read_pr","args":{"number":1},"id":"pr","type":"tool_call"}])
    model = Model(response, reply())
    store, service, session = make(model)
    monkeypatch.setattr("src.chat.runtime.read_pr", lambda *a: {"title":"fix", "url":"https://github.com/Owner/Repo/pull/1"})
    send(store, service, session)
    done = send(store, service, session, "v1", "answer")
    messages = messages_from_dict(done.runtime_messages)
    assert {m.tool_call_id for m in messages if isinstance(m, ToolMessage)} == {"ask", "pr"}

def test_search_keeps_rerank_and_repository_filter():
    class Retriever:
        def search(self, **kwargs): return [{"id":"other:1","title":"wrong"},{"id":"repo:2","title":"相关","body":"INFO"}]
    class Reranker:
        def rerank(self, **kwargs): return [dict(doc,rerank_score=.9) for doc in kwargs["docs"]]
    model = Model(tool("search_issues", {"query":"INFO"}), reply(citations=["repo:2","other:1"]))
    store, service, session = make(model, issue=False)
    service.retrieval_provider = lambda repo: (Retriever(),Reranker())
    result = send(store, service, session)
    assert result.messages[-1].citations == ["repo:2"]
    assert result.candidates[0].rerank_score == .9 and result.memory_case_id

def test_model_failure_keeps_message_for_retry():
    class APIConnectionError(Exception): pass
    store, service, session = make(Model(APIConnectionError()))
    failed = send(store, service, session)
    assert failed.status == "failed" and "连接中断" in failed.last_error
    service.answer_llm = Model(reply())
    service.process_turn(session.session_id, store.retry_failed_turn(session.session_id))
    assert store.get(session.session_id).status == "completed"

def test_plain_preference_does_not_create_case():
    store, service, session = make(Model(reply("以后先查 PR")), issue=False)
    result = send(store, service, session, "以后每次先查 PR")
    assert result.memory_case_id is None and result.retrieval_calls == 0


def test_memory_enqueue_failure_does_not_change_visible_answer(monkeypatch):
    store, service, session = make(Model(reply("答案保留")), issue=False)
    service.memory_store.set_auto_capture(True)
    monkeypatch.setattr(service.memory_service, "enqueue", lambda p: (_ for _ in ()).throw(OSError()))
    result = send(store, service, session, "以后每次先查 PR")
    assert result.status == "completed" and result.messages[-1].content == "答案保留"
    assert result.memory_organization.status == "failed"


def test_every_completed_turn_organizes_automatically(tmp_path):
    memory = MemoryStore(tmp_path / "memory")
    organizer = AIMessage(content=json.dumps({"preferences":[{"text":"每次先查关联 PR","explicit":True,"scope":"global","source_message_id":"placeholder","source_excerpt":"以后每次先查关联 PR"}]}))
    model = Model(reply("记下了"), organizer)
    store, service, session = make(model, memory=memory, issue=False)
    memory.set_auto_capture(True)
    mid, _ = store.add_user_message(session.session_id, "以后每次先查关联 PR", "first")
    model.responses[1] = AIMessage(content=organizer.content.replace("placeholder", mid))
    service.process_turn(session.session_id, mid)
    service.memory_service.wait_for_idle()
    assert memory.list()[0].status == "active"
    assert len(model.requests) == 2
    model.responses.append(reply("先查 PR"))
    memory.set_auto_capture(False)
    second = send(store, service, session, "下一次调查", "second")
    # 本会话首轮快照保持不变；后台新记忆由下一次新会话加载。
    assert second.selected_memory_context["preferences"] == []
    new_session = store.create("repo")
    model.responses.append(reply("先查 PR"))
    fresh = send(store, service, new_session, "下一次调查", "fresh")
    assert fresh.selected_memory_context["preferences"][0]["text"] == "每次先查关联 PR"


def test_actual_plain_text_reply_to_resolution_is_visible():
    text = "好的，问题解决了就行。简单收个尾：\n\n- 你的 INFO 日志问题对应 Issue #352。"
    store, service, session = make(Model(AIMessage(content=text)))
    result = send(store, service, session, "OK我解决了")
    assert result.status == "completed" and result.messages[-1].content == text
    assert result.final_response_mode == "plain_text" and result.final_response_error is None


def test_invalid_final_output_can_retry_without_fake_answer():
    store, service, session = make(Model(AIMessage(content='{"answer":"cut off')))
    result = send(store, service, session)
    assert result.status == "failed" and result.final_response_mode == "invalid_json"
    assert result.messages[-1].role == "user"
    service.answer_llm = Model(reply("回答完成"))
    service.process_turn(session.session_id, store.retry_failed_turn(session.session_id))
    assert store.get(session.session_id).status == "completed"


def test_memory_snapshot_stays_stable_while_user_correction_is_visible(monkeypatch):
    model = Model(reply(), reply())
    store, service, session = make(model, issue=False)
    reads = []
    memory = {"preferences": [{"memory_id": "p1", "text": "先查 Issue"}],
              "prior_experiences": [], "cases": []}
    monkeypatch.setattr(service.memory_service, "read_context", lambda *args: reads.append(args) or memory)
    first = send(store, service, session, "查安装错误", "one")
    # 模拟数据库侧已有内容更新；不能通过引用修改会话快照。
    memory["preferences"][0]["text"] = "先查 PR"
    second = send(store, service, session, "纠正一下，这次先查 PR，不是先查 Issue", "two")
    assert len(reads) == 1
    assert second.session_memory_snapshot == first.session_memory_snapshot
    assert second.session_memory_snapshot["preferences"][0]["text"] == "先查 Issue"
    assert model.requests[0][1].content == model.requests[1][1].content
    assert "这次先查 PR" in model.requests[1][-1].content
    assert "memory" not in second.runtime_memory_snapshot


def test_empty_memory_snapshot_is_not_reloaded(monkeypatch):
    store, service, session = make(Model(reply(), reply()), issue=False)
    reads = []
    monkeypatch.setattr(service.memory_service, "read_context", lambda *args: reads.append(args) or {})
    send(store, service, session)
    result = send(store, service, session, "再说一下", "two")
    assert len(reads) == 1 and result.session_memory_snapshot == {}


def test_memory_snapshot_survives_service_restart(tmp_path, monkeypatch):
    db = tmp_path / "sessions.db"
    store, service, session = make(Model(reply()), ChatStore(db_path=db), issue=False)
    memory = {"preferences": [{"text": "只给 PowerShell"}], "prior_experiences": [], "cases": []}
    monkeypatch.setattr(service.memory_service, "read_context", lambda *args: memory)
    send(store, service, session)
    restored = ChatStore(db_path=db)
    model = Model(reply())
    resumed_service = ChatService(store=restored, answer_llm=model, memory_store=service.memory_store)
    monkeypatch.setattr(resumed_service.memory_service, "read_context", lambda *args: pytest.fail("不能重新选记忆"))
    result = send(restored, resumed_service, session, "继续", "two")
    assert result.session_memory_snapshot == memory
    assert "只给 PowerShell" in model.requests[0][1].content


def test_legacy_session_reuses_previously_selected_memory(monkeypatch):
    store, service, session = make(Model(reply()), issue=False)
    legacy = {"preferences": [{"text": "旧会话偏好"}], "prior_experiences": [], "cases": []}
    store.update(session.session_id, lambda s: setattr(s, "selected_memory_context", legacy))
    monkeypatch.setattr(service.memory_service, "read_context", lambda *args: pytest.fail("旧会话已有读取视图"))
    result = send(store, service, session)
    assert result.session_memory_snapshot == legacy


def test_paused_question_reuses_same_memory_prefix(monkeypatch):
    model = Model(tool("ask_user", {"question": "版本？"}), reply())
    store, service, session = make(model, issue=False)
    reads = []
    monkeypatch.setattr(service.memory_service, "read_context", lambda *args: reads.append(args) or {})
    send(store, service, session)
    result = send(store, service, session, "v2", "two")
    assert len(reads) == 1 and result.status == "completed"
    assert model.requests[0][1].content == model.requests[1][1].content
