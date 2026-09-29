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


def test_retry_failed_turn_reuses_original_message_after_restart(tmp_path):
    db_path = tmp_path / "chat.sqlite3"
    first = ChatStore(db_path=db_path)
    session_id = first.create("vscode").session_id
    message_id, _ = first.add_user_message(session_id, "无法登录", "request-1")
    first.update(session_id, lambda session: (
        setattr(session, "status", "failed"),
        setattr(session, "last_error", "连接中断"),
        setattr(session, "last_search_fingerprint", "old-query"),
    ))
    second = ChatStore(db_path=db_path)
    assert second.retry_failed_turn(session_id) == message_id
    restored = second.get(session_id)
    assert restored.status == "thinking"
    assert restored.last_error is None
    assert restored.last_search_fingerprint is None
    assert [message.id for message in restored.messages] == [message_id]
    try:
        second.retry_failed_turn(session_id)
    except RuntimeError:
        pass
    else:
        assert False, "正在运行的轮次不应重复提交"


def test_interrupted_issue_publish_is_not_repeated_after_restart(tmp_path):
    path = tmp_path / "chat.sqlite3"
    store = ChatStore(db_path=path)
    session_id = store.create("gh-repo").session_id
    draft = store.save_issue_draft(session_id, "Crash on startup", "Steps").issue_draft
    store.claim_issue_publish(session_id, draft.draft_id, draft.version)

    restored = ChatStore(db_path=path)
    assert restored.get(session_id).issue_draft.status == "uncertain"
    try:
        restored.claim_issue_publish(session_id, draft.draft_id, draft.version)
    except RuntimeError:
        pass
    else:
        assert False, "网络结果未知时不能再次创建 Issue"
