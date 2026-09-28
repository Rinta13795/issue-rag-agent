"""离线替身验证：追问不重检索、补充新事实才重检索。"""

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


def test_followup_reuses_evidence_and_new_fact_retrieves_once():
    store = ChatStore()
    session = store.create("vscode")
    planner = FakeLLM([
        {"action": "retrieve", "facts": [{"value": "登录失败", "source_excerpt": "登录失败"}]},
        {"action": "reply", "reply": "候选 101 也描述登录返回 401，但仍需核对触发条件。", "citations": ["vscode:101"], "focus_candidate_id": "vscode:101"},
        {"action": "retrieve", "facts": [{"value": "v2.1", "source_excerpt": "v2.1"}, {"value": "401", "source_excerpt": "401"}]},
    ])
    answer = FakeLLM([
        {"answer": "找到一条登录问题的历史线索。", "citations": ["vscode:101"]},
        {"answer": "加入版本和错误码后，仍需核对候选细节。", "citations": ["vscode:101"]},
    ])
    retriever = FakeRetriever()
    service = ChatService(store, planner, answer, lambda repository_id: (retriever, FakeReranker()))
    first = send(service, store, session.session_id, "登录失败", "one")
    assert first.status == "completed" and first.retrieval_calls == 1
    assert first.messages[-1].citations == ["vscode:101"]
    second = send(service, store, session.session_id, "第一条为什么像？", "two")
    assert second.retrieval_calls == 1
    assert second.focus_candidate_id == "vscode:101"
    third = send(service, store, session.session_id, "补充：v2.1，返回 401", "three")
    assert third.retrieval_calls == 2
    assert retriever.queries == ["登录失败", "登录失败 v2.1 401"]
    assert third.model_calls == 5
    assert third.prompt_tokens == 50


def test_hallucinated_fact_is_rejected_but_user_symptom_can_be_searched():
    store = ChatStore()
    session = store.create("vscode")
    planner = FakeLLM([{"action": "retrieve", "facts": [{"value": "OAuth token", "source_excerpt": "OAuth token"}]}])
    retriever = FakeRetriever()
    answer = FakeLLM([{"answer": "候选提到了登录失败，但还不能确定原因。", "citations": ["vscode:101"]}])
    service = ChatService(store, planner, answer, lambda repository_id: (retriever, FakeReranker()))
    result = send(service, store, session.session_id, "登录不了", "one")
    assert result.facts == []
    assert result.retrieval_calls == 1
    assert retriever.queries == ["登录不了"]
    assert result.messages[-1].action == "retrieve"


def test_vague_symptom_searches_before_asking_for_error_code():
    store = ChatStore()
    session = store.create("vscode")
    planner = FakeLLM([{"action": "clarify", "facts": [], "open_question": "具体错误码是什么？"}])
    answer = FakeLLM([{"answer": "查到一条可能相关的历史 Issue，但不能确认相同。", "citations": ["vscode:101"]}])
    retriever = FakeRetriever()
    service = ChatService(store, planner, answer, lambda repository_id: (retriever, FakeReranker()))
    result = send(service, store, session.session_id, "我自己用的时候 skill 无法执行啊，怎么办", "one")
    assert result.retrieval_calls == 1
    assert retriever.queries == ["我自己用的时候 skill 无法执行啊，怎么办"]
    assert result.messages[-1].action == "retrieve"


def test_followup_about_solution_does_not_repeat_failed_search_or_error_code_question():
    store = ChatStore()
    session = store.create("vscode")
    planner = FakeLLM([
        {"action": "clarify", "facts": [], "open_question": "请提供错误码。"},
        {"action": "clarify", "facts": [], "open_question": "请提供错误码。"},
    ])
    retriever = FakeRetriever([])
    service = ChatService(store, planner, FakeLLM([]), lambda repository_id: (retriever, FakeReranker()))
    first = send(service, store, session.session_id, "skill 无法执行", "one")
    assert first.retrieval_calls == 1
    assert "vscode 的本地历史 Issue" in first.messages[-1].content
    second = send(service, store, session.session_id, "你自己去看啊，目前有人解决吗？", "two")
    assert second.retrieval_calls == 1
    assert second.messages[-1].action == "reply"
    assert "已经用这些线索查过" in second.messages[-1].content


def test_solution_question_uses_existing_candidates_when_planner_wants_to_clarify():
    store = ChatStore()
    session = store.create("vscode")
    planner = FakeLLM([
        {"action": "clarify", "facts": [], "open_question": "请提供错误码。"},
        {"action": "clarify", "facts": [], "open_question": "请提供错误码。"},
    ])
    answer = FakeLLM([
        {"answer": "候选 101 描述了类似问题，但未说明已修复。", "citations": ["vscode:101"]},
        {"answer": "候选 101 没有给出已解决的证据。", "citations": ["vscode:101"]},
    ])
    retriever = FakeRetriever()
    service = ChatService(store, planner, answer, lambda repository_id: (retriever, FakeReranker()))
    send(service, store, session.session_id, "skill 无法执行", "one")
    result = send(service, store, session.session_id, "目前有人解决吗？", "two")
    assert result.retrieval_calls == 1
    assert answer.calls == 2
    assert result.messages[-1].action == "reply"
    assert result.messages[-1].citations == ["vscode:101"]


def test_unrelated_candidates_do_not_become_claimed_evidence():
    store = ChatStore()
    session = store.create("vscode")
    planner = FakeLLM([{"action": "clarify", "facts": [], "open_question": "错误码是什么？"}])
    answer = FakeLLM([{"answer": "没有足够证据。", "citations": []}])
    service = ChatService(store, planner, answer, lambda repository_id: (FakeRetriever(), FakeReranker()))
    result = send(service, store, session.session_id, "skill 无法执行", "one")
    assert result.retrieval_calls == 1
    assert result.messages[-1].action == "clarify"
    assert result.messages[-1].citations == []
    assert "不足以确认" in result.messages[-1].content


def test_repeated_clarification_explains_capability_boundary_once():
    store = ChatStore()
    session = store.create("vscode")
    planner = FakeLLM([
        {"action": "clarify", "facts": [], "open_question": "请提供错误码。"},
        {"action": "clarify", "facts": [], "open_question": "请提供错误码。"},
    ])
    service = ChatService(store, planner, FakeLLM([]), lambda repository_id: (FakeRetriever(), FakeReranker()))
    first = send(service, store, session.session_id, "我想了解这个系统", "one")
    assert first.messages[-1].action == "clarify"
    second = send(service, store, session.session_id, "你自己去看不行吗？", "two")
    assert second.messages[-1].action == "reply"
    assert "不能直接读取你本机" in second.messages[-1].content
    assert second.retrieval_calls == 0


def test_missing_candidate_does_not_invent_duplicate():
    store = ChatStore()
    session = store.create("vscode")
    planner = FakeLLM([{"action": "retrieve", "facts": [{"value": "401", "source_excerpt": "401"}]}])
    retriever = FakeRetriever([])
    service = ChatService(store, planner, FakeLLM([]), lambda repository_id: (retriever, FakeReranker()))
    result = send(service, store, session.session_id, "返回 401", "one")
    assert result.retrieval_calls == 1
    assert result.messages[-1].citations == []
    assert "没有找到" in result.messages[-1].content


def test_ambiguous_reference_does_not_call_model():
    store = ChatStore()
    session = store.create("vscode")
    store.update(session.session_id, lambda state: setattr(state, "candidates", [
        ChatCandidate(id="vscode:101"), ChatCandidate(id="vscode:102"),
    ]))
    planner = FakeLLM([])
    service = ChatService(store, planner, FakeLLM([]), lambda repository_id: (FakeRetriever(), FakeReranker()))
    result = send(service, store, session.session_id, "那这个呢？", "one")
    assert planner.calls == 0
    assert result.messages[-1].action == "clarify"


def test_long_environment_only_searches_extracted_symptom():
    store = ChatStore()
    session = store.create("vscode")
    text = "Environment:\n" + ("dependency 1.0\n" * 400) + "Problem: training freezes after restart"
    planner = FakeLLM([{"action": "retrieve", "facts": [{
        "value": "training freezes after restart",
        "source_excerpt": "training freezes after restart",
    }]}])
    answer = FakeLLM([{"answer": "找到一条相关线索。", "citations": ["vscode:101"]}])
    retriever = FakeRetriever()
    service = ChatService(store, planner, answer, lambda repository_id: (retriever, FakeReranker()))
    result = send(service, store, session.session_id, text, "long-1")
    assert retriever.queries == ["training freezes after restart"]
    assert result.messages[0].content == text


def test_model_failure_is_visible_and_next_turn_is_allowed():
    class AuthenticationError(Exception):
        pass

    class FailingLLM:
        def invoke(self, messages):
            raise AuthenticationError()

    store = ChatStore()
    session = store.create("vscode")
    service = ChatService(store, FailingLLM(), FakeLLM([]), lambda repository_id: (FakeRetriever(), FakeReranker()))
    failed = send(service, store, session.session_id, "登录失败", "one")
    assert failed.status == "failed"
    assert "鉴权失败" in failed.last_error
    assert failed.messages[-1].role == "user"
    service.planner_llm = FakeLLM([{"action": "clarify", "reply": "请提供具体错误码。"}])
    resumed = send(service, store, session.session_id, "还有错误码吗？", "two")
    assert resumed.status == "completed"
    assert resumed.messages[-1].role == "assistant"


def test_chat_drops_cross_repository_candidates_even_if_retriever_misbehaves():
    store = ChatStore()
    session = store.create("openharness")
    retriever = FakeRetriever([
        {"id": "vscode:1", "title": "unrelated", "body": "wrong repository"},
        {"id": "openharness:292", "title": "skills没法执行", "body": "skill execution failed"},
    ])
    planner = FakeLLM([{"action": "retrieve", "facts": [{"value": "skill 无法执行", "source_excerpt": "skill 无法执行"}]}])
    answer = FakeLLM([{"answer": "OpenHarness 有一条相关报告。", "citations": ["vscode:1", "openharness:292"]}])
    service = ChatService(store, planner, answer, lambda repository_id: (retriever, FakeReranker()))

    result = send(service, store, session.session_id, "skill 无法执行", "one")

    assert [candidate.id for candidate in result.candidates] == ["openharness:292"]
    assert result.messages[-1].citations == ["openharness:292"]
