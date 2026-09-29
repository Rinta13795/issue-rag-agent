"""本地会话 API 的显式数据契约。"""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from config import CHAT_MAX_MESSAGE_CHARS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


ChatStatus = Literal["idle", "thinking", "retrieving", "answering", "completed", "failed"]


class ChatFact(BaseModel):
    value: str
    source_message_id: str
    source_excerpt: str


class ChatCandidate(BaseModel):
    id: str
    title: str = ""
    body_snippet: str = ""
    rerank_score: float | None = None


class ChatMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: str = Field(default_factory=utc_now)
    citations: list[str] = Field(default_factory=list)
    action: str | None = None


class ChatSession(BaseModel):
    session_id: str
    repository_id: str | None = None  # 旧会话没有仓库归属，不允许继续跨库检索
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    status: ChatStatus = "idle"
    messages: list[ChatMessage] = Field(default_factory=list)
    facts: list[ChatFact] = Field(default_factory=list)
    open_question: str | None = None
    focus_candidate_id: str | None = None
    candidates: list[ChatCandidate] = Field(default_factory=list)
    last_search_fingerprint: str | None = None
    model_calls: int = 0
    retrieval_calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    last_elapsed_ms: int | None = None
    last_error: str | None = None


class CreateChatSessionRequest(BaseModel):
    repository_id: str = Field(min_length=1, max_length=80)


class ChatSessionSummary(BaseModel):
    session_id: str
    repository_id: str | None
    title: str
    updated_at: str
    status: ChatStatus


class ChatRepository(BaseModel):
    id: str
    label: str
    issue_count: int
    source: str
    github_url: str | None = None


class SyncRepositoryRequest(BaseModel):
    repository: str = Field(min_length=3, max_length=100)


class SyncRepositoryStatus(BaseModel):
    job_id: str
    repository: str
    status: Literal["queued", "fetching", "indexing", "completed", "failed"]
    message: str
    repository_id: str | None = None


class SendChatMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=CHAT_MAX_MESSAGE_CHARS)
    client_message_id: str = Field(min_length=1, max_length=100)


class SendChatMessageResponse(BaseModel):
    message_id: str
    session_id: str
    status: ChatStatus
