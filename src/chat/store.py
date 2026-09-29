"""本地单进程聊天状态仓库；正式入口使用 SQLite 保存会话。"""

import json
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

from config import CHAT_MAX_MESSAGES, CHAT_MAX_SESSIONS, CHAT_SESSION_TTL_SECONDS
from src.chat.models import ChatMessage, ChatSession, ChatSessionSummary, IssueDraft, MemoryProposal, SourceIssue, utc_now


class ChatStore:
    def __init__(
        self,
        max_sessions: int = CHAT_MAX_SESSIONS,
        ttl_seconds: int = CHAT_SESSION_TTL_SECONDS,
        db_path: str | Path | None = None,
    ):
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._sessions: OrderedDict[str, tuple[float, ChatSession]] = OrderedDict()
        self._client_ids: dict[str, dict[str, tuple[str, str]]] = {}
        self._db: sqlite3.Connection | None = None
        if db_path is not None:
            path = Path(db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS chat_sessions ("
                "session_id TEXT PRIMARY KEY, last_seen REAL NOT NULL, "
                "payload TEXT NOT NULL, client_ids TEXT NOT NULL)"
            )
            self._db.commit()
            self._restore()

    def _restore(self) -> None:
        assert self._db is not None
        now = time.time()
        rows = self._db.execute(
            "SELECT session_id, last_seen, payload, client_ids "
            "FROM chat_sessions ORDER BY last_seen"
        ).fetchall()
        for session_id, last_seen, payload, client_ids in rows:
            if now - last_seen > self.ttl_seconds:
                self._db.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
                continue
            session = ChatSession.model_validate_json(payload)
            if session.status in ("thinking", "retrieving", "answering"):
                session.status = "failed"
                session.last_error = "服务重启中断了上一轮分析，请重新发送或补充问题。"
            if session.issue_draft and session.issue_draft.status == "publishing":
                # 写入请求可能已抵达 GitHub；重启后不能假定它没有成功。
                session.issue_draft.status = "uncertain"
            self._sessions[session_id] = (last_seen, session)
            self._client_ids[session_id] = {
                key: tuple(value) for key, value in json.loads(client_ids).items()
            }
            self._persist(session_id)
        while len(self._sessions) > self.max_sessions:
            self._remove(next(iter(self._sessions)))
        self._db.commit()

    def _persist(self, session_id: str) -> None:
        if self._db is None:
            return
        last_seen, session = self._sessions[session_id]
        self._db.execute(
            "INSERT OR REPLACE INTO chat_sessions VALUES (?, ?, ?, ?)",
            (session_id, last_seen, session.model_dump_json(), json.dumps(self._client_ids[session_id])),
        )
        self._db.commit()

    def _remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._client_ids.pop(session_id, None)
        if self._db is not None:
            self._db.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
            self._db.commit()

    def _cleanup(self) -> None:
        now = time.time()
        expired = [key for key, (last_seen, _) in self._sessions.items() if now - last_seen > self.ttl_seconds]
        for key in expired:
            self._remove(key)

    def create(self, repository_id: str | None = None, source_issue: SourceIssue | None = None) -> ChatSession:
        with self._lock:
            self._cleanup()
            while len(self._sessions) >= self.max_sessions:
                old_id = next(iter(self._sessions))
                self._remove(old_id)
            session = ChatSession(session_id=f"chat_{uuid.uuid4().hex[:16]}", repository_id=repository_id, source_issue=source_issue)
            self._sessions[session.session_id] = (time.time(), session)
            self._client_ids[session.session_id] = {}
            self._persist(session.session_id)
            return session.model_copy(deep=True)

    def list_sessions(self) -> list[ChatSessionSummary]:
        with self._lock:
            self._cleanup()
            return [ChatSessionSummary(
                session_id=session.session_id,
                repository_id=session.repository_id,
                title=next((m.content.strip().splitlines()[0][:50] for m in session.messages if m.role == "user"),
                           session.source_issue.title[:50] if session.source_issue else "新对话"),
                updated_at=session.updated_at,
                status=session.status,
            ) for _, session in sorted(self._sessions.values(), key=lambda item: item[1].updated_at, reverse=True)]

    def get(self, session_id: str) -> ChatSession | None:
        with self._lock:
            self._cleanup()
            item = self._sessions.get(session_id)
            if item is None:
                return None
            self._sessions[session_id] = (time.time(), item[1])
            self._persist(session_id)
            return item[1].model_copy(deep=True)

    def add_user_message(self, session_id: str, content: str, client_message_id: str) -> tuple[str, bool]:
        with self._lock:
            self._cleanup()
            item = self._sessions.get(session_id)
            if item is None:
                raise KeyError(session_id)
            existing = self._client_ids[session_id].get(client_message_id)
            if existing is not None:
                if existing[1] != content:
                    raise ValueError("同一 client_message_id 不能对应不同消息")
                return existing[0], False
            session = item[1]
            if session.status in ("thinking", "retrieving", "answering"):
                raise RuntimeError("会话正在处理上一条消息")
            message_id = f"msg_{uuid.uuid4().hex[:16]}"
            session.messages.append(ChatMessage(id=message_id, role="user", content=content))
            session.messages = session.messages[-CHAT_MAX_MESSAGES:]
            session.status = "thinking"
            session.last_error = None
            session.updated_at = utc_now()
            self._client_ids[session_id][client_message_id] = (message_id, content)
            self._sessions[session_id] = (time.time(), session)
            self._persist(session_id)
            return message_id, True

    def retry_failed_turn(self, session_id: str) -> str:
        """重新处理最后一条失败消息，不追加用户消息，也不丢失原对话。"""
        with self._lock:
            self._cleanup()
            item = self._sessions.get(session_id)
            if item is None:
                raise KeyError(session_id)
            session = item[1]
            if session.status != "failed" or not session.messages or session.messages[-1].role != "user":
                raise RuntimeError("当前没有可重试的失败消息")
            message_id = session.messages[-1].id
            session.status = "thinking"
            session.last_error = None
            # 失败可能发生在回答阶段；重试时允许重新检索，而不是误判为“已查过”。
            session.last_search_fingerprint = None
            session.updated_at = utc_now()
            self._sessions[session_id] = (time.time(), session)
            self._persist(session_id)
            return message_id

    def update(self, session_id: str, update_fn: Callable[[ChatSession], None]) -> ChatSession:
        with self._lock:
            item = self._sessions.get(session_id)
            if item is None:
                raise KeyError(session_id)
            session = item[1]
            update_fn(session)
            session.updated_at = utc_now()
            self._sessions[session_id] = (time.time(), session)
            self._persist(session_id)
            return session.model_copy(deep=True)

    def save_issue_draft(self, session_id: str, title: str, body: str) -> ChatSession:
        """保存模型生成的可编辑草稿；新草稿替换未发布的旧草稿。"""
        def apply(session: ChatSession):
            if session.issue_draft and session.issue_draft.status in ("publishing", "uncertain", "published"):
                raise RuntimeError("上一份草稿已发布或状态待核验；请新建对话处理另一条 Issue")
            session.issue_draft = IssueDraft(draft_id=f"draft_{uuid.uuid4().hex[:16]}", title=title, body=body)

        return self.update(session_id, apply)

    def edit_issue_draft(self, session_id: str, title: str, body: str, version: int) -> ChatSession:
        """只允许编辑未发布的当前版本，避免旧页面覆盖新草稿。"""
        def apply(session: ChatSession):
            draft = session.issue_draft
            if draft is None or draft.status != "draft" or draft.version != version:
                raise RuntimeError("草稿已变化或不可编辑，请刷新后重试")
            draft.title, draft.body, draft.version = title.strip(), body.strip(), version + 1

        return self.update(session_id, apply)

    def claim_issue_publish(self, session_id: str, draft_id: str, version: int) -> ChatSession:
        """在 SQLite 持久化发布占用状态，阻止双击并发创建。"""
        def apply(session: ChatSession):
            draft = session.issue_draft
            if draft is None or draft.draft_id != draft_id or draft.version != version:
                raise RuntimeError("草稿版本已变化，请刷新后确认")
            if draft.status == "published":
                return
            if draft.status != "draft":
                raise RuntimeError("草稿正在发布或结果不确定，不能重复提交")
            draft.status = "publishing"

        return self.update(session_id, apply)

    def finish_issue_publish(self, session_id: str, draft_id: str, url: str | None, number: int | None, uncertain: bool = False) -> ChatSession:
        """保存 GitHub 写入结果；未知超时保留待核验状态，不自动重试。"""
        def apply(session: ChatSession):
            draft = session.issue_draft
            if draft is None or draft.draft_id != draft_id or draft.status != "publishing":
                raise RuntimeError("发布中的草稿不存在")
            draft.status = "uncertain" if uncertain else ("published" if url else "draft")
            draft.published_url, draft.published_number = url, number

        return self.update(session_id, apply)

    def take_memory_proposal(self, session_id: str) -> tuple[MemoryProposal, str | None]:
        """原子取走待确认提案，避免双击确认造成两条长期记忆。"""
        with self._lock:
            item = self._sessions.get(session_id)
            if item is None:
                raise KeyError(session_id)
            session = item[1]
            if session.memory_proposal is None:
                raise RuntimeError("当前没有待确认的记忆提案")
            proposal = session.memory_proposal.model_copy(deep=True)
            session.memory_proposal = None
            session.updated_at = utc_now()
            self._sessions[session_id] = (time.time(), session)
            self._persist(session_id)
            return proposal, session.repository_id

    def restore_memory_proposal(self, session_id: str, proposal: MemoryProposal) -> None:
        """长期记忆写入失败时恢复用户可核对的提案。"""
        self.update(session_id, lambda session: setattr(session, "memory_proposal", session.memory_proposal or proposal))


_CHAT_STORE: ChatStore | None = None


def get_chat_store() -> ChatStore:
    global _CHAT_STORE
    if _CHAT_STORE is None:
        project_root = Path(__file__).resolve().parents[2]
        _CHAT_STORE = ChatStore(db_path=project_root / ".local" / "chat_sessions.sqlite3")
    return _CHAT_STORE
