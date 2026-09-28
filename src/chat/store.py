"""进程内、单会话串行的聊天状态仓库。"""

import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable

from config import CHAT_MAX_MESSAGES, CHAT_MAX_SESSIONS, CHAT_SESSION_TTL_SECONDS
from src.chat.models import ChatMessage, ChatSession, utc_now


class ChatStore:
    def __init__(self, max_sessions: int = CHAT_MAX_SESSIONS, ttl_seconds: int = CHAT_SESSION_TTL_SECONDS):
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._sessions: OrderedDict[str, tuple[float, ChatSession]] = OrderedDict()
        self._client_ids: dict[str, dict[str, tuple[str, str]]] = {}

    def _cleanup(self) -> None:
        now = time.monotonic()
        expired = [key for key, (last_seen, _) in self._sessions.items() if now - last_seen > self.ttl_seconds]
        for key in expired:
            self._sessions.pop(key, None)
            self._client_ids.pop(key, None)

    def create(self) -> ChatSession:
        with self._lock:
            self._cleanup()
            while len(self._sessions) >= self.max_sessions:
                old_id, _ = self._sessions.popitem(last=False)
                self._client_ids.pop(old_id, None)
            session = ChatSession(session_id=f"chat_{uuid.uuid4().hex[:16]}")
            self._sessions[session.session_id] = (time.monotonic(), session)
            self._client_ids[session.session_id] = {}
            return session.model_copy(deep=True)

    def get(self, session_id: str) -> ChatSession | None:
        with self._lock:
            self._cleanup()
            item = self._sessions.get(session_id)
            if item is None:
                return None
            self._sessions[session_id] = (time.monotonic(), item[1])
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
            self._sessions[session_id] = (time.monotonic(), session)
            return message_id, True

    def update(self, session_id: str, update_fn: Callable[[ChatSession], None]) -> ChatSession:
        with self._lock:
            item = self._sessions.get(session_id)
            if item is None:
                raise KeyError(session_id)
            session = item[1]
            update_fn(session)
            session.updated_at = utc_now()
            self._sessions[session_id] = (time.monotonic(), session)
            return session.model_copy(deep=True)


_CHAT_STORE: ChatStore | None = None


def get_chat_store() -> ChatStore:
    global _CHAT_STORE
    if _CHAT_STORE is None:
        _CHAT_STORE = ChatStore()
    return _CHAT_STORE
