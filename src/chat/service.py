"""聊天协调层：调查 Runtime、草稿与后台记忆入口。"""

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from config import (
    CHAT_ANSWER_MAX_TOKENS,
    CHAT_MAX_MESSAGES,
    CHAT_PLANNER_MAX_TOKENS,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LLM_TIMEOUT,
)
from src.chat.context import (
    candidate_from_doc,
    parse_json_object,
    source_issue_payload,
    select_message_excerpt,
)
from src.chat.models import ChatMessage, ChatSession
from src.chat.models import MemoryProposal
from src.chat.memory import MemoryStore, get_memory_store
from src.chat.memory_service import MemoryService
from src.chat.memory_prompt import memory_prompt_payload
from src.chat.usage import record_usage
from src.chat.models import MemoryCase, MemoryOrganizationStatus
from src.chat.github_sync import search_live_issues
from src.chat.repositories import list_repositories
from src.chat.prompts import ISSUE_DRAFT_SYSTEM, MEMORY_ORGANIZER_SYSTEM, MEMORY_SYSTEM
from src.chat.store import ChatStore, get_chat_store


def _default_retrieval_provider(repository_id: str):
    from src.chat.repositories import get_chat_retrieval_dependencies

    return get_chat_retrieval_dependencies(repository_id)


class ChatService:
    def __init__(
        self,
        store: ChatStore | None = None,
        planner_llm: Any | None = None,
        answer_llm: Any | None = None,
        retrieval_provider: Callable | None = None,
        memory_store: MemoryStore | None = None,
    ):
        self.store = store or get_chat_store()
        self.planner_llm = planner_llm
        self.answer_llm = answer_llm
        self.retrieval_provider = retrieval_provider or _default_retrieval_provider
        self.memory_store = memory_store or (get_memory_store() if store is None else MemoryStore())
        self.memory_service = MemoryService(self.memory_store, self._extract_memory, self._memory_status_changed)

    def _llm(self, kind: str):
        existing = self.planner_llm if kind == "planner" else self.answer_llm
        if existing is not None:
            return existing
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("未配置模型服务")
        client = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.1,
            max_tokens=CHAT_PLANNER_MAX_TOKENS if kind == "planner" else CHAT_ANSWER_MAX_TOKENS,
            timeout=LLM_TIMEOUT,
            max_retries=0,
            stream_usage=True,
        )
        if kind == "planner":
            self.planner_llm = client
        else:
            self.answer_llm = client
        return client

    def _call(self, session_id: str, kind: str, system: str, payload: str) -> dict:
        def count_call(session: ChatSession):
            session.model_calls += 1

        self.store.update(session_id, count_call)
        response = self._llm(kind).invoke([SystemMessage(content=system), HumanMessage(content=payload)])
        self.store.update(session_id, lambda s: record_usage(s, response))
        return parse_json_object(response.content)

    def _extract_memory(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._llm("answer").invoke([
            SystemMessage(content=MEMORY_ORGANIZER_SYSTEM),
            HumanMessage(content=json.dumps(memory_prompt_payload(payload), ensure_ascii=False, separators=(",", ":"))),
        ])
        session_id = payload.get("session_id")
        if session_id:
            def update(session: ChatSession):
                session.model_calls += 1
                record_usage(session, response)
            try:
                self.store.update(session_id, update)
            except KeyError:
                pass
        return parse_json_object(response.content)

    def _memory_status_changed(self, payload: dict[str, Any], status: str, error: str | None) -> None:
        session_id = payload.get("session_id")
        if not session_id:
            return
        try:
            def update(session: ChatSession):
                session.memory_organization = MemoryOrganizationStatus(
                    enabled=self.memory_store.auto_capture, status=status, last_error=error,
                )
                case_id = payload.get("case_id")
                case = self.memory_store.get_case(case_id) if case_id else None
                if case and status == "completed":
                    session.investigation_summary = case.summary
            self.store.update(session_id, update)
        except KeyError:
            pass

    def _ensure_case(self, session_id: str, current: ChatMessage) -> MemoryCase:
        snapshot = self.store.get(session_id)
        if snapshot is None or snapshot.repository_id is None:
            raise KeyError(session_id)
        issue = snapshot.source_issue
        source_issue_id = f"{issue.repository}#{issue.number}" if issue else None
        case = self.memory_store.get_case(snapshot.memory_case_id) if snapshot.memory_case_id else None
        if case is None:
            case = self.memory_store.find_case(snapshot.repository_id, source_issue_id, session_id)
        if case is None:
            title = issue.title if issue else current.content.strip().splitlines()[0][:120]
            case = MemoryCase(
                case_id=f"case_{uuid.uuid4().hex[:16]}", repository_id=snapshot.repository_id,
                title=title or "未命名问题", source_issue_id=source_issue_id,
                source_issue_url=issue.url if issue else None, session_ids=[session_id],
            )
        else:
            case = case.model_copy(update={"session_ids": list(dict.fromkeys(case.session_ids + [session_id]))})
        case = self.memory_store.create_or_update_case(case)
        self.store.update(session_id, lambda session: (
            setattr(session, "memory_case_id", case.case_id),
            setattr(session, "investigation_summary", case.summary),
        ))
        return case

    def _finish(self, session_id: str, content: str, action: str, citations: list[str], question: str | None):
        def update(session: ChatSession):
            session.messages.append(ChatMessage(
                id=f"msg_{uuid.uuid4().hex[:16]}", role="assistant", content=content,
                citations=citations, action=action,
            ))
            session.messages = session.messages[-CHAT_MAX_MESSAGES:]
            session.streaming_answer = ""
            session.streaming_turn_id = None
            session.open_question = question
            session.status = "completed"

        self.store.update(session_id, update)
        if self.memory_store.auto_capture:
            self._queue_memory_organization(session_id)

    def _queue_memory_organization(self, session_id: str) -> None:
        try:
            self._queue_memory_snapshot(session_id)
        except Exception:
            self.store.update(session_id, lambda s: setattr(s, "memory_organization",
                MemoryOrganizationStatus(enabled=True, status="failed", last_error="整理任务未能保存；回答已保留")))

    def _queue_memory_snapshot(self, session_id: str) -> None:
        snapshot = self.store.get(session_id)
        if snapshot is None or snapshot.repository_id is None:
            return
        user_messages = [message for message in snapshot.messages if message.role == "user"]
        assistant_messages = [message for message in snapshot.messages if message.role == "assistant"]
        if not user_messages or not assistant_messages:
            return
        current_user = user_messages[-1]
        final_answer = assistant_messages[-1]
        self.store.update(session_id, lambda session: setattr(
            session, "memory_organization", MemoryOrganizationStatus(enabled=True, status="queued"),
        ))
        payload = {
            "session_id": session_id,
            "message_id": current_user.id,
            "repository_id": snapshot.repository_id,
            "case_id": snapshot.memory_case_id,
            "investigation_summary": snapshot.investigation_summary,
            "current_tool_evidence_ids": [step.result.get("evidence_id") for step in snapshot.runtime_steps
                                          if step.turn_id == current_user.id and step.result.get("evidence_id")],
            "tool_evidence": [item.model_dump() for item in snapshot.evidence[-12:]],
            "existing_records": [item.model_dump() for item in self.memory_store.list(snapshot.repository_id)
                                 if item.case_id == snapshot.memory_case_id or item.kind == "preference"][:30],
            "user_messages": [{"id": message.id, "content": select_message_excerpt(message.content, 600),
                               "created_at": message.created_at} for message in user_messages[-6:]],
            "assistant_message": {"id": final_answer.id, "content": final_answer.content[:1200],
                                  "citations": final_answer.citations, "action": final_answer.action},
            "source_issue": ({
                "repository": snapshot.source_issue.repository, "number": snapshot.source_issue.number,
                "title": snapshot.source_issue.title[:240], "body": select_message_excerpt(snapshot.source_issue.body, 2200),
                "url": snapshot.source_issue.url,
            } if snapshot.source_issue else None),
            "candidates": [{
                "id": candidate.id, "title": candidate.title[:240],
                "body_snippet": candidate.body_snippet[:700], "url": candidate.url,
            } for candidate in snapshot.candidates[:3]],
        }
        self.memory_service.enqueue(payload)

    def create_draft(self, session_id: str) -> ChatSession:
        """从当前问题的有限上下文起草，实时查重仅提供线索，绝不发布。"""
        snapshot = self.store.get(session_id)
        if snapshot is None:
            raise KeyError(session_id)
        if not snapshot.source_issue and not any(message.role == "user" for message in snapshot.messages):
            raise ValueError("请先导入 Issue 或描述要反馈的问题")
        payload = json.dumps({
            "repository_id": snapshot.repository_id,
            "source_issue": source_issue_payload(snapshot, 3000),
            "user_facts": [fact.value for fact in snapshot.facts],
            "recent_user_messages": [select_message_excerpt(message.content, 1200) for message in snapshot.messages if message.role == "user"][-5:],
            "related_candidates": [candidate.model_dump() for candidate in snapshot.candidates],
        }, ensure_ascii=False)
        parsed = self._call(session_id, "answer", ISSUE_DRAFT_SYSTEM, payload)
        title = parsed.get("title") if isinstance(parsed.get("title"), str) else ""
        body = parsed.get("body") if isinstance(parsed.get("body"), str) else ""
        title, body = title.strip()[:256], body.strip()[:20000]
        if not title or not body:
            raise ValueError("模型没有生成有效草稿；请补充问题描述后重试")
        saved = self.store.save_issue_draft(session_id, title, body)
        repo = next((item for item in list_repositories() if item.id == saved.repository_id), None)
        if repo and repo.github_url:
            try:
                live_docs = search_live_issues(repo.github_url.removeprefix("https://github.com/"), title, limit=5)
                if saved.source_issue:
                    live_docs = [doc for doc in live_docs if doc["id"] != f"{saved.repository_id}:{saved.source_issue.number}"]
                suggestions = [candidate_from_doc(doc) for doc in live_docs[:5]]
                return self.store.update(session_id, lambda session: (
                    setattr(session.issue_draft, "possible_duplicates", suggestions),
                    setattr(session.issue_draft, "search_status", "ok"),
                ))
            except Exception:
                return self.store.update(session_id, lambda session: setattr(session.issue_draft, "search_status", "failed"))
        return saved

    def create_memory_proposal(self, session_id: str) -> ChatSession:
        """仅在用户要求时提炼候选记忆，校验原文来源，等待用户确认。"""
        snapshot = self.store.get(session_id)
        if snapshot is None:
            raise KeyError(session_id)
        if not snapshot.messages and not snapshot.source_issue:
            raise ValueError("请先处理一个 Issue 或描述你的处理习惯")
        sources = [message.content for message in snapshot.messages if message.role == "user"]
        if snapshot.source_issue:
            sources.extend([snapshot.source_issue.title, snapshot.source_issue.body])
        payload = json.dumps({
            "repository_id": snapshot.repository_id,
            "source_issue": source_issue_payload(snapshot),
            "recent_dialogue": [{"role": message.role, "text": message.content[:800]} for message in snapshot.messages[-10:]],
            "published_issue_url": snapshot.issue_draft.published_url if snapshot.issue_draft else None,
        }, ensure_ascii=False)
        raw = self._call(session_id, "answer", MEMORY_SYSTEM, payload)
        text = raw.get("text") if isinstance(raw.get("text"), str) else ""
        excerpt = raw.get("source_excerpt") if isinstance(raw.get("source_excerpt"), str) else ""
        text, excerpt = text.strip()[:500], excerpt.strip()[:240]
        if not text or not excerpt or not any(excerpt in source for source in sources):
            raise ValueError("这段对话还没有足够明确、可核对的经验；不会自动保存")
        kind = raw.get("kind") if raw.get("kind") in ("preference", "experience") else "experience"
        scope = "repository" if kind == "experience" else ("global" if raw.get("scope") == "global" else "repository")
        proposal = MemoryProposal(kind=kind, scope=scope, text=text, source_excerpt=excerpt)
        return self.store.update(session_id, lambda session: setattr(session, "memory_proposal", proposal))

    def process_turn(self, session_id: str, message_id: str) -> None:
        start = time.monotonic()
        try:
            snapshot = self.store.get(session_id)
            if snapshot is None:
                return
            if snapshot.repository_id is None:
                self._finish(session_id, "这是升级前未绑定仓库的旧对话。请新建对话并选择仓库，避免跨仓库匹配。", "reply", [], None)
                return
            from src.chat.runtime import InvestigationRuntime
            InvestigationRuntime(self).run(session_id, message_id)
        except Exception as exc:
            error_type = type(exc).__name__
            cause_type = type(exc.__cause__).__name__ if exc.__cause__ else "none"
            logger.warning("聊天任务失败 session_id={} error_type={} cause_type={}", session_id, error_type, cause_type)
            if error_type in ("AuthenticationError", "PermissionDeniedError"):
                public_error = "模型服务拒绝了请求。请检查本地服务配置；这条消息已保留。"
            elif "ConnectionError" in error_type or error_type == "APIConnectionError":
                public_error = "模型服务连接中断。这条消息已保留，可以重试本轮。"
            elif "Timeout" in error_type:
                public_error = "模型服务响应超时。这条消息已保留，可以重试本轮。"
            elif getattr(exc, "status_code", None) == 402:
                public_error = "模型服务额度不足。这条消息已保留。"
            else:
                public_error = "本轮分析没有完成。这条消息已保留，可以重试本轮。"
            try:
                self.store.update(session_id, lambda session: (
                    setattr(session, "streaming_answer", ""),
                    setattr(session, "streaming_turn_id", None),
                    setattr(session, "status", "failed"),
                    setattr(session, "last_error", public_error),
                ))
            except KeyError:
                pass
        finally:
            try:
                elapsed = int((time.monotonic() - start) * 1000)
                self.store.update(session_id, lambda session: setattr(session, "last_elapsed_ms", elapsed))
            except KeyError:
                pass


class ChatRunner:
    def __init__(self, service: ChatService | None = None):
        self.service = service or ChatService()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-chat-worker")

    def submit(self, session_id: str, message_id: str) -> None:
        self.executor.submit(self.service.process_turn, session_id, message_id)


_CHAT_RUNNER: ChatRunner | None = None


def get_chat_runner() -> ChatRunner:
    global _CHAT_RUNNER
    if _CHAT_RUNNER is None:
        _CHAT_RUNNER = ChatRunner()
    return _CHAT_RUNNER
