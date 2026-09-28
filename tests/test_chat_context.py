"""对话上下文过滤不接受模型虚构的用户事实。"""

import json

from src.chat.context import (
    is_ambiguous_reference,
    merge_facts,
    parse_json_object,
    planner_payload,
    search_query,
    select_message_excerpt,
    validate_plan,
)
from src.chat.models import ChatCandidate
from src.chat.store import ChatStore


def new_turn(text: str):
    store = ChatStore()
    session = store.create()
    message_id, _ = store.add_user_message(session.session_id, text, "client-1")
    return store.get(session.session_id), message_id


def test_short_fact_must_be_in_user_message():
    session, message_id = new_turn("登录失败")
    plan = validate_plan({
        "action": "retrieve", "facts": [
            {"value": "登录失败", "source_excerpt": "登录失败"},
            {"value": "OAuth token 过期", "source_excerpt": "OAuth token 过期"},
        ],
    }, session, message_id)
    assert [fact.value for fact in plan.facts] == ["登录失败"]
    assert search_query(plan.facts) == "登录失败"


def test_long_log_context_is_bounded_and_keeps_signal():
    text = "环境配置\n" + ("dependency = 1.0\n" * 700) + "Problem: app crashes after restart\n"
    session, message_id = new_turn(text)
    payload = planner_payload(session, message_id)
    assert "app crashes after restart" in payload
    assert len(json.loads(payload)["current_message"]) <= 4000
    assert session.messages[0].content == text


def test_ambiguous_reference_needs_clarification():
    session, _ = new_turn("这个呢？")
    session.candidates = [ChatCandidate(id="101"), ChatCandidate(id="102")]
    assert is_ambiguous_reference("那这个呢？", session)
    session.focus_candidate_id = "101"
    assert not is_ambiguous_reference("那这个呢？", session)


def test_candidate_id_whitelist_and_deduplicated_facts():
    session, message_id = new_turn("v2.1 返回 401")
    session.candidates = [ChatCandidate(id="101")]
    plan = validate_plan({
        "action": "reply", "focus_candidate_id": "999", "citations": ["999", "101", "101"],
        "facts": [{"value": "401", "source_excerpt": "返回 401"}],
    }, session, message_id)
    assert plan.focus_candidate_id is None
    assert plan.citations == ["101"]
    assert len(merge_facts(plan.facts, plan.facts)) == 1


def test_invalid_json_and_empty_facts_are_safe():
    session, message_id = new_turn("哦")
    assert parse_json_object("not json") == {}
    plan = validate_plan({}, session, message_id)
    assert plan.action == "clarify"
    assert plan.facts == []


def test_excerpt_for_short_message_is_verbatim():
    assert select_message_excerpt("NullPointerException") == "NullPointerException"
