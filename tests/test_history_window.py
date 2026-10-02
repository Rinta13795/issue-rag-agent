"""历史窗口策略单测：默认滑动窗口与原行为一致，只追加类策略只在截断/压缩时改变前缀。"""

from src.chat.history import SUMMARY_PREFIX, select_history
from src.chat.models import ChatMessage, ChatSession


def _session(turns: int) -> ChatSession:
    messages = []
    for i in range(1, turns + 1):
        messages.append(ChatMessage(id=f"u{i}", role="user", content=f"用户第{i}轮"))
        messages.append(ChatMessage(id=f"a{i}", role="assistant", content=f"助手第{i}轮"))
    messages.append(ChatMessage(id="cur", role="user", content="当前问题"))
    return ChatSession(session_id="s", messages=messages)


def _texts(history):
    return [m.content for m in history]


def test_sliding_matches_previous_last_ten_including_current():
    session = _session(12)
    history, updates = select_history(session, "cur", "sliding")
    assert updates == {}
    assert _texts(history) == [m.content for m in session.messages[-10:-1]]


def test_full_keeps_everything_before_current():
    session = _session(12)
    history, _ = select_history(session, "cur", "full")
    assert len(history) == 24


def test_stepped_grows_until_limit_then_cuts_back_once():
    session = _session(9)  # 18 条历史 + 当前 = 19，未超过 20
    history, updates = select_history(session, "cur", "stepped")
    assert len(history) == 18 and updates == {}
    session = _session(10)  # 20 条历史 + 当前 = 21，砍回最近 10 条（含当前）
    history, updates = select_history(session, "cur", "stepped")
    assert len(history) == 9 and updates == {"history_window_start": "a6"}
    session.history_window_start = "a6"
    session.messages.insert(-1, ChatMessage(id="u11", role="user", content="追加"))
    history, updates = select_history(session, "cur", "stepped")
    assert len(history) == 10 and updates == {}  # 截断后只追加，前缀不再变化


def test_compact_summarizes_old_part_and_keeps_recent_six():
    session = _session(9)  # 18 条未压缩历史 > 16
    calls = []
    history, updates = select_history(session, "cur", "compact", lambda prev, old: calls.append(len(old)) or "摘要")
    assert calls == [12]
    assert history[0].content == SUMMARY_PREFIX + "摘要" and len(history) == 7
    assert updates == {"history_summary": "摘要", "history_summary_through": "a6"}
    session.history_summary, session.history_summary_through = "摘要", "a6"
    history, updates = select_history(session, "cur", "compact", lambda prev, old: "不应调用")
    assert updates == {} and len(history) == 7  # 未超过阈值时只追加，不重复压缩


def test_case_summary_freezes_snapshot_and_skips_when_empty():
    session = _session(9)
    history, updates = select_history(session, "cur", "case_summary")
    assert updates == {} and len(history) == 18  # 案例摘要为空时不压缩
    session.investigation_summary = "案例摘要 v1"
    history, updates = select_history(session, "cur", "case_summary")
    assert updates["history_summary"] == "案例摘要 v1" and len(history) == 7
