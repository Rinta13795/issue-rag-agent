"""与短期会话分离的、用户确认后才写入的 Issue 处理经验。"""

from __future__ import annotations

import re
import sqlite3
import threading
import uuid
from pathlib import Path

from src.chat.models import MemoryProposal, MemoryRecord


def _bigrams(value: str) -> set[str]:
    """用轻量字符二元组匹配中英文问题，不额外加载向量模型。"""
    text = re.sub(r"\s+", " ", value.casefold()).strip()
    return {text[index:index + 2] for index in range(max(len(text) - 1, 0))}


class MemoryStore:
    def __init__(self, db_path: str | Path | None = None):
        self._lock = threading.RLock()
        self._records: dict[str, MemoryRecord] = {}
        self._db = None
        if db_path is not None:
            path = Path(db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS chat_memories (memory_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            self._db.commit()
            for memory_id, payload in self._db.execute("SELECT memory_id, payload FROM chat_memories"):
                self._records[memory_id] = MemoryRecord.model_validate_json(payload)

    def add(self, proposal: MemoryProposal, repository_id: str | None, session_id: str) -> MemoryRecord:
        """将已确认提案独立持久化，仓库经验必须带仓库 ID。"""
        scoped_id = repository_id if proposal.scope == "repository" else None
        if proposal.scope == "repository" and not scoped_id:
            raise ValueError("仓库记忆缺少仓库归属")
        record = MemoryRecord(**proposal.model_dump(), memory_id=f"mem_{uuid.uuid4().hex[:16]}",
                              repository_id=scoped_id, source_session_id=session_id)
        with self._lock:
            self._records[record.memory_id] = record
            if self._db:
                self._db.execute("INSERT INTO chat_memories VALUES (?, ?)", (record.memory_id, record.model_dump_json()))
                self._db.commit()
        return record

    def list(self, repository_id: str | None = None) -> list[MemoryRecord]:
        """返回当前用户的本地记忆；可按仓库缩小查看范围。"""
        with self._lock:
            records = list(self._records.values())
        if repository_id:
            records = [item for item in records if item.repository_id in (None, repository_id)]
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    def delete(self, memory_id: str) -> bool:
        """用户主动撤销记忆后，后续对话不再加载它。"""
        with self._lock:
            if memory_id not in self._records:
                return False
            del self._records[memory_id]
            if self._db:
                self._db.execute("DELETE FROM chat_memories WHERE memory_id = ?", (memory_id,))
                self._db.commit()
            return True

    def select(self, repository_id: str | None, query: str, limit: int = 3) -> list[MemoryRecord]:
        """全局偏好优先，仓库经验按当前问题相似度加载，控制上下文预算。"""
        relevant = self.list(repository_id)
        preferences = [item for item in relevant if item.kind == "preference"][:2]
        query_grams = _bigrams(query)
        experiences = [item for item in relevant if item.kind == "experience" and item.repository_id == repository_id]
        scored = sorted(((len(_bigrams(item.text) & query_grams), item) for item in experiences),
                        key=lambda pair: pair[0], reverse=True)
        return (preferences + [item for score, item in scored if score > 0])[:limit]


_MEMORY_STORE: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    global _MEMORY_STORE
    if _MEMORY_STORE is None:
        root = Path(__file__).resolve().parents[2]
        _MEMORY_STORE = MemoryStore(root / ".local" / "chat_memories.sqlite3")
    return _MEMORY_STORE
