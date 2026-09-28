"""本地会话在进程重启后的恢复语义。"""

from src.chat.store import ChatStore


def test_chat_session_and_idempotency_survive_store_restart(tmp_path):
    db_path = tmp_path / "chat.sqlite3"
    first = ChatStore(db_path=db_path)
    session_id = first.create().session_id
    message_id, created = first.add_user_message(session_id, "登录失败", "request-1")
    assert created
    first.update(session_id, lambda session: setattr(session, "status", "completed"))

    second = ChatStore(db_path=db_path)
    restored = second.get(session_id)
    assert restored is not None
    assert restored.messages[0].content == "登录失败"
    assert second.add_user_message(session_id, "登录失败", "request-1") == (message_id, False)


def test_interrupted_turn_is_recoverable_after_restart(tmp_path):
    db_path = tmp_path / "chat.sqlite3"
    first = ChatStore(db_path=db_path)
    session_id = first.create().session_id
    first.add_user_message(session_id, "返回 401", "request-1")

    second = ChatStore(db_path=db_path)
    restored = second.get(session_id)
    assert restored is not None
    assert restored.status == "failed"
    assert "重启" in restored.last_error
    _, created = second.add_user_message(session_id, "补充：v2.1", "request-2")
    assert created
