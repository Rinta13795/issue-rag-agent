"""本地会话 API 的显式数据契约。"""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from config import CHAT_MAX_MESSAGE_CHARS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


ChatStatus = Literal["idle", "thinking", "retrieving", "answering", "completed", "failed", "waiting_for_user"]


class ChatFact(BaseModel):
    value: str
    source_message_id: str
    source_excerpt: str


class ChatCandidate(BaseModel):
    id: str
    title: str = ""
    body_snippet: str = ""
    rerank_score: float | None = None
    source: Literal["local", "github"] = "local"
    url: str | None = None


class SourceIssueComment(BaseModel):
    author: str
    body: str
    url: str


class SourceIssue(BaseModel):
    repository: str
    number: int
    title: str
    body: str
    state: str
    url: str
    comments: list[SourceIssueComment] = Field(default_factory=list)
    comments_truncated: bool = False
    linked_prs: list[dict] = Field(default_factory=list)
    links_truncated: bool = False
    links_error: str | None = None


class IssueDraft(BaseModel):
    draft_id: str
    title: str
    body: str
    version: int = 1
    status: Literal["draft", "publishing", "published", "uncertain"] = "draft"
    published_url: str | None = None
    published_number: int | None = None
    possible_duplicates: list[ChatCandidate] = Field(default_factory=list)
    search_status: Literal["not_run", "ok", "failed"] = "not_run"


class MemoryProposal(BaseModel):
    kind: Literal["preference", "experience"]
    text: str
    scope: Literal["global", "repository"]
    source_excerpt: str


class MemorySourceRef(BaseModel):
    source_type: Literal["user_message", "source_issue", "candidate", "published_issue", "tool_evidence"]
    source_id: str
    excerpt: str
    url: str | None = None
    created_at: str = Field(default_factory=utc_now)


class MemoryRevision(BaseModel):
    revision: int
    text: str
    status: str
    changed_at: str = Field(default_factory=utc_now)
    changed_by: Literal["automatic", "user"] = "automatic"
    reason: str | None = None


class MemoryRecord(MemoryProposal):
    memory_id: str
    repository_id: str | None
    source_session_id: str
    created_at: str = Field(default_factory=utc_now)
    status: Literal["observed", "active", "pending", "supported", "verified", "refuted"] = "pending"
    verification_scope: Literal["unspecified", "problem", "solution"] = "unspecified"
    origin: Literal["explicit", "inferred", "user", "assistant", "legacy"] = "user"
    entry_type: Literal["preference", "observation", "plan", "hypothesis", "attempt", "result", "summary"] = "observation"
    case_id: str | None = None
    source_refs: list[MemorySourceRef] = Field(default_factory=list)
    supporting_sessions: list[str] = Field(default_factory=list)
    revision: int = 1
    revision_history: list[MemoryRevision] = Field(default_factory=list)
    modified_by: Literal["automatic", "user"] = "user"


class MemoryCase(BaseModel):
    case_id: str
    repository_id: str
    title: str
    summary: str = ""
    source_issue_url: str | None = None
    source_issue_id: str | None = None
    session_ids: list[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=utc_now)


class MemoryOrganizationStatus(BaseModel):
    enabled: bool = True
    status: Literal["idle", "queued", "processing", "completed", "failed"] = "idle"
    last_error: str | None = None


class ChatMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: str = Field(default_factory=utc_now)
    citations: list[str] = Field(default_factory=list)
    action: str | None = None


class InvestigationEvidence(BaseModel):
    id: str
    kind: Literal["issue", "comment", "pr", "diff", "search"]
    title: str = ""
    text: str = ""
    url: str | None = None
    fetched_at: str = Field(default_factory=utc_now)
    truncated: bool = False
    metadata: dict = Field(default_factory=dict)


class RuntimeStep(BaseModel):
    call_id: str
    turn_id: str
    tool: str
    arguments: dict = Field(default_factory=dict)
    status: Literal["running", "completed", "failed", "waiting", "cancelled"] = "running"
    result: dict = Field(default_factory=dict)


class PendingQuestion(BaseModel):
    question_id: str
    call_id: str
    question: str
    options: list[str] = Field(default_factory=list)
    turn_id: str


class AnswerQuestionRequest(BaseModel):
    question_id: str = Field(min_length=1, max_length=100)
    client_message_id: str = Field(min_length=1, max_length=100)
    answer: str = Field(default="", max_length=CHAT_MAX_MESSAGE_CHARS)
    cancelled: bool = False


class ChatSession(BaseModel):
    session_id: str
    repository_id: str | None = None  # 旧会话没有仓库归属，不允许继续跨库检索
    source_issue: SourceIssue | None = None
    pending_question: PendingQuestion | None = None
    runtime_messages: list[dict] = Field(default_factory=list)
    runtime_steps: list[RuntimeStep] = Field(default_factory=list)
    evidence: list[InvestigationEvidence] = Field(default_factory=list)
    final_response_mode: str | None = None
    final_response_error: str | None = None
    runtime_resume: bool = False
    runtime_cancelled: bool = False
    runtime_turn_id: str | None = None
    runtime_memory_snapshot: dict = Field(default_factory=dict)
    runtime_memory_ids: list[str] = Field(default_factory=list)
    issue_draft: IssueDraft | None = None
    memory_proposal: MemoryProposal | None = None
    memory_case_id: str | None = None
    investigation_summary: str = ""
    memory_organization: MemoryOrganizationStatus = Field(default_factory=MemoryOrganizationStatus)
    selected_memory_context: dict[str, list[dict]] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    status: ChatStatus = "idle"
    messages: list[ChatMessage] = Field(default_factory=list)
    facts: list[ChatFact] = Field(default_factory=list)
    open_question: str | None = None
    focus_candidate_id: str | None = None
    candidates: list[ChatCandidate] = Field(default_factory=list)
    last_search_fingerprint: str | None = None
    live_search_status: Literal["not_run", "ok", "failed"] = "not_run"
    live_search_message: str | None = None
    model_calls: int = 0
    retrieval_calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    last_elapsed_ms: int | None = None
    last_error: str | None = None


class CreateChatSessionRequest(BaseModel):
    repository_id: str = Field(min_length=1, max_length=80)
    issue_url: str | None = Field(default=None, max_length=300)
    case_id: str | None = Field(default=None, max_length=80)


class PreviewIssueRequest(BaseModel):
    issue_url: str = Field(min_length=1, max_length=300)


class UpdateIssueDraftRequest(BaseModel):
    title: str = Field(min_length=1, max_length=256)
    body: str = Field(min_length=1, max_length=20000)
    version: int = Field(ge=1)


class PublishIssueDraftRequest(BaseModel):
    draft_id: str = Field(min_length=1, max_length=80)
    version: int = Field(ge=1)
    confirmed: bool


class ConfirmMemoryRequest(BaseModel):
    kind: Literal["preference", "experience"]
    text: str = Field(min_length=1, max_length=500)
    scope: Literal["global", "repository"]
    confirmed: bool


class UpdateMemoryRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    status: Literal["observed", "active", "pending", "supported", "verified", "refuted"] | None = None


class MemorySettings(BaseModel):
    auto_capture: bool = True


class UpdateMemorySettingsRequest(BaseModel):
    auto_capture: bool


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
