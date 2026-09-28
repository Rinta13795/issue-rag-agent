"""聊天协调层：先过滤上下文，必要时检索，最后基于候选证据回答。"""

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
    CHAT_MAX_CANDIDATES,
    CHAT_MAX_MESSAGES,
    CHAT_PLANNER_MAX_TOKENS,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    HYBRID_TOP_K,
    LLM_TIMEOUT,
)
from src.chat.context import (
    candidate_from_doc,
    fallback_search_query,
    is_ambiguous_reference,
    merge_facts,
    parse_json_object,
    planner_payload,
    search_fingerprint,
    search_query,
    select_message_excerpt,
    validate_plan,
)
from src.chat.models import ChatCandidate, ChatFact, ChatMessage, ChatSession
from src.chat.prompts import ANSWER_SYSTEM, PLANNER_SYSTEM
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
    ):
        self.store = store or get_chat_store()
        self.planner_llm = planner_llm
        self.answer_llm = answer_llm
        self.retrieval_provider = retrieval_provider or _default_retrieval_provider

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
        usage = getattr(response, "usage_metadata", None) or {}
        if usage:
            def count_tokens(session: ChatSession):
                input_tokens = usage.get("input_tokens")
                output_tokens = usage.get("output_tokens")
                if isinstance(input_tokens, int):
                    session.prompt_tokens = (session.prompt_tokens or 0) + input_tokens
                if isinstance(output_tokens, int):
                    session.completion_tokens = (session.completion_tokens or 0) + output_tokens

            self.store.update(session_id, count_tokens)
        return parse_json_object(response.content)

    def _finish(self, session_id: str, content: str, action: str, citations: list[str], question: str | None):
        def update(session: ChatSession):
            session.messages.append(ChatMessage(
                id=f"msg_{uuid.uuid4().hex[:16]}", role="assistant", content=content,
                citations=citations, action=action,
            ))
            session.messages = session.messages[-CHAT_MAX_MESSAGES:]
            session.open_question = question
            session.status = "completed"

        self.store.update(session_id, update)

    def _answer_from_candidates(self, session_id: str, current: ChatMessage, facts: list[ChatFact], candidates: list[ChatCandidate], query: str, action: str) -> None:
        self.store.update(session_id, lambda session: setattr(session, "status", "answering"))
        payload = json.dumps({
            "selected_repository": self.store.get(session_id).repository_id,
            "current_message": select_message_excerpt(current.content, 1800),
            "user_facts": [fact.value for fact in facts],
            "search_query": query,
            "candidates": [candidate.model_dump() for candidate in candidates],
        }, ensure_ascii=False)
        answer = self._call(session_id, "answer", ANSWER_SYSTEM, payload)
        allowed = {candidate.id for candidate in candidates}
        citations = list(dict.fromkeys(str(item) for item in answer.get("citations", []) if str(item) in allowed)) if isinstance(answer.get("citations"), list) else []
        body = answer.get("answer") if isinstance(answer.get("answer"), str) else ""
        question = answer.get("open_question") if isinstance(answer.get("open_question"), str) else None
        if not body.strip() or not citations:
            body = "我查了本地历史 Issue，但现有候选还不足以确认是同一故障或已有可靠解法。你可以查看候选原文，或补充报错原文与仓库名称。"
            citations = []
            if action == "retrieve":
                action = "clarify"
                question = "能提供报错原文或仓库名称吗？"
        self._finish(session_id, body[:1600], action, citations, question[:240] if question else None)

    def process_turn(self, session_id: str, message_id: str) -> None:
        start = time.monotonic()
        try:
            snapshot = self.store.get(session_id)
            if snapshot is None:
                return
            if snapshot.repository_id is None:
                self._finish(session_id, "这是升级前未绑定仓库的旧对话。请新建对话并选择仓库，避免跨仓库匹配。", "reply", [], None)
                return
            current = next(message for message in snapshot.messages if message.id == message_id)
            if is_ambiguous_reference(current.content, snapshot):
                self._finish(session_id, "你指的是上面哪条历史 Issue？可以说第一条、第二条，或给我候选 ID。", "clarify", [], "你指的是哪条历史 Issue？")
                return
            raw = self._call(session_id, "planner", PLANNER_SYSTEM, planner_payload(snapshot, message_id))
            plan = validate_plan(raw, snapshot, message_id)
            facts = merge_facts(snapshot.facts, plan.facts)

            def apply_plan(session: ChatSession):
                session.facts = facts
                if plan.focus_candidate_id:
                    session.focus_candidate_id = plan.focus_candidate_id

            self.store.update(session_id, apply_plan)
            query = search_query(facts)
            fallback_query = fallback_search_query(snapshot, message_id)
            if not query and fallback_query:
                query = fallback_query
            fingerprint = search_fingerprint(query) if query else None
            should_retrieve = plan.action == "retrieve" or (
                bool(fallback_query) and not snapshot.candidates
            )

            if should_retrieve and query and (
                plan.force_search or fingerprint != snapshot.last_search_fingerprint
            ):
                self.store.update(session_id, lambda session: setattr(session, "status", "retrieving"))
                retriever, reranker = self.retrieval_provider(snapshot.repository_id)
                docs = retriever.search(query=query, top_k=HYBRID_TOP_K, project=snapshot.repository_id)
                docs = [doc for doc in docs if str(doc.get("id", "")).startswith(snapshot.repository_id + ":")]
                ranked = reranker.rerank(query=query, docs=docs)
                candidates = [candidate_from_doc(doc) for doc in ranked if doc.get("id") and (doc.get("title") or doc.get("body"))]
                candidates = candidates[:CHAT_MAX_CANDIDATES]

                def record_search(session: ChatSession):
                    session.retrieval_calls += 1
                    session.last_search_fingerprint = fingerprint
                    session.candidates = candidates
                    session.focus_candidate_id = None

                self.store.update(session_id, record_search)
                if not candidates:
                    self._finish(session_id, f"我在 {snapshot.repository_id} 的本地历史 Issue 中没有找到可核对的候选。这不代表最新 Issue 没有人解决。你可以补充报错原文，或在新对话中选择其他仓库。", "clarify", [], "能补充报错原文或复现步骤吗？")
                    return
                self._answer_from_candidates(session_id, current, facts, candidates, query, "retrieve")
                return

            # 没有事实变化时，沿用当前证据而不是再次检索。
            if plan.action == "retrieve" and not query:
                body = "目前还缺少能检索的故障现象。你遇到了什么错误，发生在哪一步？"
                self._finish(session_id, body, "clarify", [], body)
                return
            if should_retrieve and fingerprint == snapshot.last_search_fingerprint:
                if snapshot.candidates and any(marker in current.content for marker in ("解决", "修复", "有人处理")):
                    self._answer_from_candidates(session_id, current, facts, snapshot.candidates, query, "reply")
                    return
                body = "我已经用这些线索查过本地历史 Issue，没有新的线索时重复检索不会改变结果。你可以补充报错原文，或查看已有候选。"
                self._finish(session_id, body, "reply", [], None)
                return
            if snapshot.candidates and plan.action == "clarify" and any(marker in current.content for marker in ("解决", "修复", "有人处理")):
                self._answer_from_candidates(session_id, current, facts, snapshot.candidates, query, "reply")
                return
            allowed = {candidate.id for candidate in snapshot.candidates}
            citations = [issue_id for issue_id in plan.citations if issue_id in allowed]
            candidate_question = any(marker in current.content for marker in ("第一条", "第二条", "第三条", "候选", "Issue", "issue", "为什么像", "怎么解决"))
            body = plan.reply
            if plan.action == "clarify" or not body:
                body = plan.open_question or "我还不确定你指的是哪条记录。能说一下候选编号，或补充具体错误提示吗？"
                if any(message.role == "assistant" and message.action == "clarify" for message in snapshot.messages[-2:]):
                    body = "我不能直接读取你本机的终端或日志，只能查询当前本地历史 Issue 库。现有信息还不足以确认原因；你可以补充报错原文，也可以让我先按已有现象搜索。"
                    self._finish(session_id, body, "reply", citations, None)
                    return
            elif candidate_question and snapshot.candidates and not citations:
                body = "我需要确认你指的是哪条候选 Issue，才能对照已有证据解释。可以说候选编号或第几条吗？"
            self._finish(session_id, body[:1600], plan.action, citations, plan.open_question)
        except Exception as exc:
            error_type = type(exc).__name__
            logger.warning("聊天任务失败 session_id={} error_type={}", session_id, error_type)
            if error_type in ("AuthenticationError", "PermissionDeniedError"):
                public_error = "模型服务鉴权失败。请更新本地 .env 中的 DEEPSEEK_API_KEY 后重启服务。"
            elif getattr(exc, "status_code", None) == 402:
                public_error = "模型服务当前不可用，请检查账户余额或额度。"
            else:
                public_error = "当前分析未完成。请检查模型服务配置或稍后重试。"
            try:
                self.store.update(session_id, lambda session: (
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
