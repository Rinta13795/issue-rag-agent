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
    is_issue_or_pr_advice,
    merge_facts,
    parse_json_object,
    planner_payload,
    search_fingerprint,
    search_query,
    source_issue_query,
    source_issue_payload,
    select_message_excerpt,
    validate_plan,
)
from src.chat.models import ChatCandidate, ChatFact, ChatMessage, ChatSession
from src.chat.models import MemoryProposal
from src.chat.memory import MemoryStore, get_memory_store
from src.chat.github_sync import search_live_issues
from src.chat.repositories import list_repositories
from src.chat.prompts import ANSWER_SYSTEM, ISSUE_DRAFT_SYSTEM, MEMORY_SYSTEM, PLANNER_SYSTEM
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
        snapshot = self.store.get(session_id)
        payload = json.dumps({
            "selected_repository": snapshot.repository_id,
            "source_issue": source_issue_payload(snapshot),
            "current_message": select_message_excerpt(current.content, 1800),
            "user_facts": [fact.value for fact in facts],
            "search_query": query,
            "candidates": [candidate.model_dump() for candidate in candidates],
            "relevant_memories": [item.text for item in self.memory_store.select(snapshot.repository_id, current.content)],
        }, ensure_ascii=False)
        answer = self._call(session_id, "answer", ANSWER_SYSTEM, payload)
        allowed = {candidate.id for candidate in candidates}
        citations = list(dict.fromkeys(str(item) for item in answer.get("citations", []) if str(item) in allowed)) if isinstance(answer.get("citations"), list) else []
        body = answer.get("answer") if isinstance(answer.get("answer"), str) else ""
        question = answer.get("open_question") if isinstance(answer.get("open_question"), str) else None
        if not body.strip() or not citations:
            search_scope = "本地快照和 GitHub 实时搜索" if snapshot.live_search_status == "ok" else "本地 Issue 快照（实时搜索未完成）"
            if "反馈" in current.content or "类似问题" in current.content:
                body = f"我按你描述的行为查了当前仓库的{search_scope}，但这批候选没有足够证据证明已有同类反馈。这不代表仓库全部历史 Issue 都没有；你可以查看候选原文，或补充具体触发步骤再查。"
                question = None
            else:
                body = f"我查了当前仓库的{search_scope}，但现有候选还不足以确认是同一故障或已有可靠解法。你可以查看候选原文，或补充报错原文。"
                question = "能提供报错原文或触发步骤吗？"
            citations = []
            if action == "retrieve":
                action = "clarify" if question else "reply"
                if action == "clarify":
                    question = "能提供报错原文或触发步骤吗？"
        self._finish(session_id, body[:1600], action, citations, question[:240] if question else None)

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
            current = next(message for message in snapshot.messages if message.id == message_id)
            if is_issue_or_pr_advice(current.content):
                self._finish(session_id, "如果是在报告现象、体验问题或请求排查，通常先提 Issue；PR 更适合已有明确代码修改和验证结果的情况。我可以先查当前仓库有没有类似反馈，再帮你决定是关联已有 Issue，还是起草新 Issue。", "reply", [], None)
                return
            if is_ambiguous_reference(current.content, snapshot):
                self._finish(session_id, "你指的是上面哪条历史 Issue？可以说第一条、第二条，或给我候选 ID。", "clarify", [], "你指的是哪条历史 Issue？")
                return
            plan_payload = json.loads(planner_payload(snapshot, message_id))
            memory_query = current.content + " " + (snapshot.source_issue.title if snapshot.source_issue else "")
            plan_payload["relevant_memories"] = [item.text for item in self.memory_store.select(snapshot.repository_id, memory_query)]
            raw = self._call(session_id, "planner", PLANNER_SYSTEM, json.dumps(plan_payload, ensure_ascii=False))
            plan = validate_plan(raw, snapshot, message_id)
            facts = merge_facts(snapshot.facts, plan.facts)

            def apply_plan(session: ChatSession):
                session.facts = facts
                if plan.focus_candidate_id:
                    session.focus_candidate_id = plan.focus_candidate_id

            self.store.update(session_id, apply_plan)
            if plan.action == "draft_issue":
                try:
                    draft_session = self.create_draft(session_id)
                except ValueError as exc:
                    self._finish(session_id, str(exc), "clarify", [], str(exc))
                    return
                title = draft_session.issue_draft.title
                self._finish(session_id, f"我起草了一条新 Issue：{title}。请先检查下方草稿和可能重复的反馈；目前还没有发布。", "reply", [], None)
                return
            if plan.action == "propose_memory":
                try:
                    proposed = self.create_memory_proposal(session_id)
                except ValueError as exc:
                    self._finish(session_id, str(exc), "reply", [], None)
                    return
                self._finish(session_id, f"我提炼了一条待确认的经验：{proposed.memory_proposal.text}。请在下方核对或修改；现在还没有写入长期记忆。", "reply", [], None)
                return
            query = search_query(facts)
            if not query:
                query = source_issue_query(snapshot)
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
                live_status, live_message = "not_run", None
                repo = next((item for item in list_repositories() if item.id == snapshot.repository_id), None)
                if repo and repo.github_url:
                    try:
                        full_name = repo.github_url.removeprefix("https://github.com/")
                        live_docs = search_live_issues(full_name, query)
                        seen = {str(doc.get("id")) for doc in docs}
                        docs.extend(doc for doc in live_docs if doc["id"] not in seen)
                        live_status = "ok"
                    except Exception as exc:
                        logger.warning("GitHub 实时搜索失败 error_type={}", type(exc).__name__)
                        live_status, live_message = "failed", "GitHub 暂时不可用或额度受限"
                if snapshot.source_issue:
                    source_id = f"{snapshot.repository_id}:{snapshot.source_issue.number}"
                    docs = [doc for doc in docs if str(doc.get("id")) != source_id]
                ranked = reranker.rerank(query=query, docs=docs)
                candidates = [candidate_from_doc(doc) for doc in ranked if doc.get("id") and (doc.get("title") or doc.get("body"))]
                candidates = candidates[:CHAT_MAX_CANDIDATES]

                def record_search(session: ChatSession):
                    session.retrieval_calls += 1
                    session.last_search_fingerprint = fingerprint
                    session.candidates = candidates
                    session.live_search_status = live_status
                    session.live_search_message = live_message
                    session.focus_candidate_id = None

                self.store.update(session_id, record_search)
                if not candidates:
                    live_note = "GitHub 实时搜索也未返回可核对候选" if live_status == "ok" else "GitHub 实时搜索未完成"
                    self._finish(session_id, f"我在 {snapshot.repository_id} 的本地历史 Issue 中没有找到可核对候选；{live_note}。这不代表仓库里肯定没人遇过。你可以补充线索，或让我先起草一条 Issue 供你检查。", "reply", [], None)
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
                if any(marker in current.content for marker in ("你自己去看", "自己看日志", "看我终端")):
                    self._finish(session_id, "我不能直接读取你本机终端或日志；如果给我公开 GitHub Issue 链接，我可以读取那条 Issue，并查询当前仓库的历史反馈。", "reply", citations, None)
                    return
            elif candidate_question and snapshot.candidates and not citations:
                body = "我需要确认你指的是哪条候选 Issue，才能对照已有证据解释。可以说候选编号或第几条吗？"
            self._finish(session_id, body[:1600], plan.action, citations, plan.open_question)
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
