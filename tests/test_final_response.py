import pytest
from langchain_core.messages import AIMessage
from src.chat.final_response import parse_final_response, is_unspecified_resolution


def test_actual_plain_text_completion_reply_is_kept():
    text = "好的，问题解决了就行。简单收个尾：\n\n- 你的 INFO 日志问题对应 Issue #352，解法在 PR #359。"
    result = parse_final_response(AIMessage(content=text), {"pr:HKUDS/OpenHarness#359"})
    assert result.answer == text and result.mode == "plain_text" and result.citations == []


@pytest.mark.parametrize("text,mode", [('', 'invalid'), ('{"answer":"未结束', 'invalid_json'), ('{}','invalid_schema'), ('{"answer":42}','invalid_schema'), ('[]','invalid_schema')])
def test_bad_output_is_not_reported_as_insufficient_evidence(text, mode):
    result = parse_final_response(AIMessage(content=text), set())
    assert result.mode == mode and result.error and result.answer is None
    assert "证据不足" not in result.error and "补充线索" not in result.error


def test_token_limit_is_reported_and_json_citations_filtered():
    assert parse_final_response(AIMessage(content='{"answer":"text"}',response_metadata={"finish_reason":"length"}),set()).mode == "truncated"
    result = parse_final_response(AIMessage(content='{"answer":"完成", "citations":["real","fake","real",42]}'),{"real"})
    assert result.answer == "完成" and result.citations == ["real"]


@pytest.mark.parametrize("text", ["OK我解决了", "我已经解决了！", "好了", "我搞定了"])
def test_short_completion_feedback(text):
    assert is_unspecified_resolution(text)


@pytest.mark.parametrize("text", ["没有解决", "解决了吗？", "按照 PR 修改后解决了", "如果解决了就好了", "还是不行", "没好了"])
def test_other_intents_are_not_short_resolution(text):
    assert not is_unspecified_resolution(text)


@pytest.mark.parametrize("prefix", ["调查完成。以下是结论。\n```json\n", "调查完成：\n"])
def test_extracts_complete_envelope_after_prose(prefix):
    text = prefix + '{"answer":"正常正文", "citations":["real","fake"]}' + '\n```'
    result = parse_final_response(AIMessage(content=text), {"real"})
    assert result.answer == "正常正文" and result.citations == ["real"]


def test_prefixed_truncated_envelope_is_not_plain_text():
    result = parse_final_response(AIMessage(content='调查完成。\n```json\n{"answer":"正文", "citations":["broken'), set())
    assert result.answer is None and result.mode == "invalid_json"


def test_complete_object_before_a_duplicate_truncated_object():
    text = '{"answer":"正常正文", "citations":["real"]}\n```json\n{"answer":"重复", "citations":["broken'
    result = parse_final_response(AIMessage(content=text), {"real"})
    assert result.answer == "正常正文" and result.citations == ["real"]
