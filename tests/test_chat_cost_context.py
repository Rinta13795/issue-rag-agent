"""Bound memory input and record provider-reported cache usage without guessing."""
import json
from langchain_core.messages import AIMessage, SystemMessage
from src.chat.memory_prompt import MEMORY_INPUT_CHAR_BUDGET, memory_prompt_payload
from src.chat.models import ChatSession
from src.chat.usage import record_usage


def test_memory_view_keeps_current_evidence_and_does_not_mutate_sources():
    payload = {
        'session_id': 's', 'message_id': 'u', 'user_messages': [{'id': 'u', 'content': '有效了'}],
        'current_tool_evidence_ids': ['new'],
        'tool_evidence': [{'id': 'old', 'text': 'old diff' * 10000}, {'id': 'new', 'text': 'new evidence' * 10000}],
        'existing_records': [{'memory_id': 'm', 'revision': 3, 'text': '计划' * 3000, 'revision_history': ['history' * 20000]}],
    }
    result = memory_prompt_payload(payload)
    assert len(json.dumps(result, ensure_ascii=False, separators=(',', ':'))) <= MEMORY_INPUT_CHAR_BUDGET
    assert [e['id'] for e in result['tool_evidence']] == ['new']
    assert result['tool_evidence'][0]['truncated']
    assert 'revision_history' not in result['existing_records'][0]
    assert result['existing_records'][0]['revision'] == 3
    assert len(payload['tool_evidence'][1]['text']) > 100000
    assert payload['existing_records'][0]['revision_history']
    payload['current_tool_evidence_ids'] = []
    assert memory_prompt_payload(payload)['tool_evidence'] == []


def test_cache_reports_use_only_known_coverage():
    session = ChatSession(session_id='s')
    record_usage(session, AIMessage(content='', usage_metadata={'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110}))
    assert session.cached_input_tokens is None and session.cache_reported_input_tokens == 0
    record_usage(session, AIMessage(content='', usage_metadata={'input_tokens': 200, 'output_tokens': 5, 'total_tokens': 205, 'input_token_details': {'cache_read': 120}}))
    record_usage(session, AIMessage(content='', response_metadata={'token_usage': {'prompt_tokens': 100, 'completion_tokens': 5, 'prompt_cache_hit_tokens': 0}}))
    assert session.prompt_tokens == 400 and session.completion_tokens == 20
    assert session.cached_input_tokens == 120 and session.cache_reported_input_tokens == 300


def test_dynamic_context_follows_unchanged_history_and_precedes_current_user():
    from test_chat_runtime import Model, make, send, reply
    model = Model(reply('第一次'), reply('第二次'))
    store, service, session = make(model, issue=False)
    send(store, service, session, '第一次问题', 'one')
    send(store, service, session, '第二次问题', 'two')
    messages = model.requests[1]
    assert messages[1].content == '第一次问题'
    assert messages[2].content == '第一次'
    assert isinstance(messages[3], SystemMessage) and messages[3].content.startswith('调查上下文：')
    assert messages[4].content == '第二次问题'
