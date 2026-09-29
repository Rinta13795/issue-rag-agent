"""从对话原话构建有界上下文，并校验 LLM 的结构化规划。"""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from config import (
    CHAT_EVIDENCE_CHARS,
    CHAT_MAX_CANDIDATES,
    CHAT_MAX_FACTS,
    CHAT_QUERY_CHARS,
    CHAT_RECENT_MESSAGES,
)
from src.chat.models import ChatCandidate, ChatFact, ChatSession


@dataclass
class ContextPlan:
    action: str
    facts: list[ChatFact]
    focus_candidate_id: str | None
    reply: str
    citations: list[str]
    open_question: str | None
    force_search: bool = False


def parse_json_object(value: Any) -> dict:
    if not isinstance(value, str):
        return {}
    cleaned = value.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, dict) else {}
    except (ValueError, TypeError):
        return {}


def select_message_excerpt(text: str, limit: int = 4000) -> str:
    """长日志只给规划器首尾与疑似故障行；原文仍完整保存在会话。"""
    if len(text) <= limit:
        return text
    signal = re.compile(r"error|exception|traceback|fail|crash|problem|bug|freeze|报错|失败|崩溃|卡住|无法", re.I)
    lines = [line.strip()[:220] for line in text.splitlines() if signal.search(line)]
    middle = "\n".join(lines[:12])[:1400]
    return f"{text[:1100]}\n[中间内容已省略]\n{middle}\n[尾部]\n{text[-1100:]}"[:limit]


def is_ambiguous_reference(text: str, session: ChatSession) -> bool:
    """无焦点且展示多条候选时，不能擅自解释“这个”指哪条。"""
    short = re.sub(r"[\s，。？！?！]", "", text)
    return len(session.candidates) > 1 and session.focus_candidate_id is None and short in {
        "那这个呢", "这个呢", "它呢", "那它呢", "这个怎么解决", "它怎么解决",
    }


def source_issue_payload(session: ChatSession, body_chars: int = 1800) -> dict | None:
    """源 Issue 在数据库保留原文；给 LLM 只送有限首尾和评论摘录。"""
    issue = session.source_issue
    if issue is None:
        return None
    return {
        "repository": issue.repository, "number": issue.number, "title": issue.title,
        "body_excerpt": select_message_excerpt(issue.body, body_chars),
        "state": issue.state, "url": issue.url,
        "comments": [{"author": item.author, "body_excerpt": select_message_excerpt(item.body, 300), "url": item.url}
                     for item in issue.comments[-3:]],
        "comments_truncated": issue.comments_truncated or len(issue.comments) > 3,
    }


def planner_payload(session: ChatSession, current_message_id: str) -> str:
    current = next(message for message in session.messages if message.id == current_message_id)
    previous = [m for m in session.messages if m.id != current_message_id][-CHAT_RECENT_MESSAGES:]
    payload = {
        "selected_repository": session.repository_id,
        "source_issue": source_issue_payload(session),
        "current_message": select_message_excerpt(current.content),
        "known_user_facts": [fact.value for fact in session.facts[-CHAT_MAX_FACTS:]],
        "recent_dialogue": [{"role": m.role, "text": m.content[:500]} for m in previous],
        "focus_candidate_id": session.focus_candidate_id,
        "current_candidates": [candidate.model_dump() for candidate in session.candidates[:CHAT_MAX_CANDIDATES]],
    }
    return json.dumps(payload, ensure_ascii=False)


def validate_plan(raw: dict, session: ChatSession, current_message_id: str) -> ContextPlan:
    current = next(message for message in session.messages if message.id == current_message_id)
    action = raw.get("action") if raw.get("action") in ("reply", "clarify", "retrieve", "draft_issue", "propose_memory") else "clarify"
    facts: list[ChatFact] = []
    for item in raw.get("facts", []) if isinstance(raw.get("facts"), list) else []:
        if not isinstance(item, dict):
            continue
        value, excerpt = item.get("value"), item.get("source_excerpt")
        if not isinstance(value, str) or not isinstance(excerpt, str):
            continue
        value, excerpt = value.strip(), excerpt.strip()
        if not value or len(value) > 160 or len(excerpt) > 240:
            continue
        if value not in excerpt or excerpt not in current.content:
            continue
        facts.append(ChatFact(value=value, source_excerpt=excerpt, source_message_id=current_message_id))
        if len(facts) >= 4:
            break
    valid_ids = {candidate.id for candidate in session.candidates}
    focus = raw.get("focus_candidate_id")
    if not isinstance(focus, str) or focus not in valid_ids:
        focus = None
    citations = raw.get("citations") if isinstance(raw.get("citations"), list) else []
    citations = list(dict.fromkeys(str(item) for item in citations if str(item) in valid_ids))
    reply = raw.get("reply") if isinstance(raw.get("reply"), str) else ""
    question = raw.get("open_question") if isinstance(raw.get("open_question"), str) else None
    return ContextPlan(action, facts, focus, reply.strip()[:1200], citations, question[:240] if question else None, raw.get("force_search") is True)


def merge_facts(existing: list[ChatFact], added: list[ChatFact]) -> list[ChatFact]:
    seen = {fact.value.casefold() for fact in existing}
    merged = list(existing)
    for fact in added:
        if fact.value.casefold() not in seen:
            merged.append(fact)
            seen.add(fact.value.casefold())
    return merged[-CHAT_MAX_FACTS:]


def search_query(facts: list[ChatFact]) -> str:
    return " ".join(fact.value for fact in facts)[:CHAT_QUERY_CHARS].strip()


def source_issue_query(session: ChatSession) -> str:
    """已导入 Issue 时，用其真实标题和正文作为首轮检索输入。"""
    if not session.source_issue:
        return ""
    source = session.source_issue
    return f"{source.title} {source.body[:500]}"[:CHAT_QUERY_CHARS].strip()


def is_issue_or_pr_advice(text: str) -> bool:
    """过程咨询直接回答；不能误当成需要用户提供错误码的故障。"""
    return bool(re.search(r"(?i)(issue|问题单|反馈)", text) and re.search(r"(?i)\bpr\b|pull request|要不要提|需要提|该不该提", text))


_FAULT_SIGNAL = re.compile(r"error|exception|traceback|fail|crash|报错|失败|崩溃|卡住|无法|不能|不了", re.I)
_SEARCH_REQUEST = re.compile(r"查|搜|有没有.*解决|有人解决|先找|有没有人反馈|有人遇到|类似问题|反馈过", re.I)


def fallback_search_query(session: ChatSession, current_message_id: str) -> str:
    """规划器漏提事实时，从故障现象或明确的历史反馈请求中保留原话检索。"""
    current = next(message for message in session.messages if message.id == current_message_id)
    messages = [current]
    if _SEARCH_REQUEST.search(current.content):
        messages.extend(
            message for message in reversed(session.messages)
            if message.role == "user" and message.id != current_message_id
        )
    for message in messages:
        for line in message.content.splitlines():
            line = line.strip()
            if _FAULT_SIGNAL.search(line):
                return line[:min(CHAT_QUERY_CHARS, 240)]
    # 行为不符合预期同样可能是 Issue；“有没有人反馈”不是必须带错误码的技术故障。
    if _SEARCH_REQUEST.search(current.content) and len(current.content.strip()) >= 20:
        return select_message_excerpt(current.content, min(CHAT_QUERY_CHARS, 240)).strip()
    return ""


def search_fingerprint(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def candidate_from_doc(doc: dict) -> ChatCandidate:
    return ChatCandidate(
        id=str(doc.get("id", "")),
        title=str(doc.get("title", ""))[:240],
        body_snippet=str(doc.get("body", ""))[:CHAT_EVIDENCE_CHARS],
        rerank_score=float(doc["rerank_score"]) if doc.get("rerank_score") is not None else None,
        source="github" if doc.get("source") == "github" else "local",
        url=str(doc["url"]) if doc.get("url") else None,
    )
