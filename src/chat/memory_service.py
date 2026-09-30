"""Automatic memory extraction and file-backed task processing."""

from __future__ import annotations

import re
import hashlib
import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable

from src.chat.final_response import is_unspecified_resolution
from src.chat.memory import MemoryStore
from src.chat.models import MemoryCase, MemoryRecord, MemorySourceRef


_LONG_TERM_MARKERS = (
    "以后", "今后", "之后都", "每次", "总是", "长期", "我的习惯", "我习惯",
    "我偏好", "我倾向", "我喜欢", "我不喜欢", "默认", "请记住",
)


class MemoryService:
    """Reads relevant memory and asynchronously updates cases/preferences."""

    def __init__(
        self,
        store: MemoryStore,
        extractor: Callable[[dict[str, Any]], dict[str, Any]],
        on_status: Callable[[dict[str, Any], str, str | None], None] | None = None,
    ):
        self.store = store
        self.extractor = extractor
        self.on_status = on_status
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="issue-rag-memory")
        self._lock = threading.RLock()
        self._futures: set[Future] = set()
        for job in self.store.pending_jobs():
            self._submit(job)

    def _submit(self, job: dict[str, Any]) -> None:
        future = self._executor.submit(self._process, job)
        with self._lock:
            self._futures.add(future)
        future.add_done_callback(lambda done: self._discard_future(done))

    def _discard_future(self, future: Future) -> None:
        with self._lock:
            self._futures.discard(future)

    def enqueue(self, payload: dict[str, Any]) -> str | None:
        if not self.store.auto_capture:
            return None
        job_id, is_new = self.store.enqueue_job(payload)
        if is_new:
            self._submit(self.store.job(job_id) or {"job_id": job_id, "payload": payload})
        return job_id

    def wait_for_idle(self, timeout: float = 5.0) -> None:
        """Test and local diagnostics helper; production requests never wait."""
        deadline = datetime.now(timezone.utc).timestamp() + timeout
        while True:
            with self._lock:
                futures = list(self._futures)
            if not futures:
                return
            remaining = deadline - datetime.now(timezone.utc).timestamp()
            if remaining <= 0:
                return
            for future in futures:
                try:
                    future.result(timeout=remaining)
                except Exception:
                    pass

    def read_context(self, repository_id: str, query: str, case_id: str | None = None) -> dict[str, Any]:
        preferences = [record for record in self.store.select(repository_id, query, limit=6)
                       if record.kind == "preference"][:3]
        prior_experiences = [record for record in self.store.select(repository_id, query, limit=6)
                             if record.kind == "experience" and record.case_id is None][:3]
        cases = self.store.select_cases(repository_id, query, limit=3)
        current = self.store.get_case(case_id) if case_id else None
        cases = ([current] if current else []) + [case for case in cases if not current or case.case_id != current.case_id]
        cases = cases[:4]
        context = {
            "preferences": [self._record_payload(record) for record in preferences],
            "prior_experiences": [self._record_payload(record) for record in prior_experiences],
            "cases": [self._case_payload(case) for case in cases],
        }
        while len(json.dumps(context, ensure_ascii=False)) > 4000:
            reducible = next((case for case in reversed(context["cases"]) if case["entries"]), None)
            if reducible:
                reducible["entries"].pop()
            elif len(context["cases"]) > 1:
                context["cases"].pop()
            elif context["preferences"]:
                context["preferences"].pop()
            elif context["prior_experiences"]:
                context["prior_experiences"].pop()
            else:
                break
        return context

    @staticmethod
    def _record_payload(record: MemoryRecord) -> dict[str, Any]:
        return {
            "memory_id": record.memory_id,
            "revision": record.revision,
            "text": record.text[:220],
            "status": record.status,
            "verification_scope": record.verification_scope,
            "origin": record.origin,
            "scope": record.scope,
            "source": record.source_excerpt[:100],
            "sources": [{"type": source.source_type, "url": source.url} for source in record.source_refs[:2]],
            "updated_at": record.created_at,
        }

    def _case_payload(self, case: MemoryCase) -> dict[str, Any]:
        records = [record for record in self.store.list(case.repository_id) if record.case_id == case.case_id]
        return {
            "case_id": case.case_id,
            "title": case.title[:100],
            "summary": case.summary[:360],
            "updated_at": case.updated_at,
            "source_issue_url": case.source_issue_url,
            "entries": [{
                "type": record.entry_type,
                "text": record.text[:180],
                "status": record.status,
                "verification_scope": record.verification_scope,
                "source": record.source_excerpt[:100],
            } for record in records[:3]],
        }

    def _process(self, job: dict[str, Any]) -> None:
        job_id = job["job_id"]
        payload = job.get("payload") or {}
        try:
            self.store.set_job_status(job_id, "processing")
            self._report(payload, "processing", None)
            current_job = self.store.job(job_id) or {}
            if "extracted" in current_job:
                extracted = current_job["extracted"]
            else:
                extracted = self.extractor(payload)
                self.store.save_job_extraction(job_id, extracted)
            self._apply(payload, extracted)
            self.store.set_job_status(job_id, "completed")
            self._report(payload, "completed", None)
        except Exception as exc:
            message = type(exc).__name__
            self.store.set_job_status(job_id, "failed", message)
            self._report(payload, "failed", message)

    def _report(self, payload: dict[str, Any], status: str, error: str | None) -> None:
        if self.on_status:
            try:
                self.on_status(payload, status, error)
            except Exception:
                pass

    def _apply(self, payload: dict[str, Any], extracted: dict[str, Any]) -> None:
        session_id = payload["session_id"]
        repository_id = payload.get("repository_id")
        case_id = payload.get("case_id")
        user_messages = payload.get("user_messages", [])
        message_by_id = {item["id"]: item for item in user_messages if isinstance(item, dict) and item.get("id")}
        issue = payload.get("source_issue") or {}
        candidates = payload.get("candidates", [])

        current_message = message_by_id.get(str(payload.get("message_id")))
        bare_resolution = current_message and is_unspecified_resolution(current_message.get("content", ""))
        if bare_resolution and case_id:
            case = self.store.get_case(case_id)
            if case:
                report = "用户报告当前问题已解决；采用方法未说明，尚不能确认具体 PR 或方案有效。"
                previous = case.summary
                if previous.startswith(report):
                    previous = previous[len(report):].strip().removeprefix("此前调查：").strip()
                self.store.update_case_summary(case_id, report + ("\n此前调查：" + previous[:1200] if previous else ""), session_id)
                source_ref = MemorySourceRef(source_type="user_message", source_id=current_message["id"], excerpt=current_message["content"])
                prior_result = next((record for record in self.store.list(repository_id)
                    if record.case_id == case_id and record.entry_type == "result" and record.origin == "user"
                    and any(ref.source_type == "user_message" and ref.source_id == current_message["id"] for ref in record.source_refs)
                    and any(word in record.text for word in ("未说明", "未确认", "方法未知"))), None)
                report_record = MemoryRecord(
                    kind="experience", text="用户报告当前问题已解决；采用方法未说明", scope="repository",
                    source_excerpt=source_ref.excerpt, memory_id="mem_" + hashlib.sha256(
                        f"{session_id}:{payload['message_id']}:problem_resolution".encode()).hexdigest()[:16],
                    repository_id=repository_id, source_session_id=session_id, status="verified",
                    verification_scope="problem", origin="user", entry_type="result", case_id=case_id,
                    source_refs=[source_ref], supporting_sessions=[session_id], modified_by="automatic",
                )
                if prior_result:
                    self.store.revise_automatic(prior_result.memory_id, report_record, prior_result.revision)
                else:
                    self.store.add_record(report_record)
        summary = extracted.get("summary")
        if not bare_resolution and case_id and isinstance(summary, str) and summary.strip():
            self.store.update_case_summary(case_id, summary.strip(), session_id)

        case = self.store.get_case(case_id) if case_id else None
        entries = extracted.get("entries") if isinstance(extracted.get("entries"), list) else []
        for entry in entries[:8]:
            if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
                continue
            text = entry["text"].strip()[:500]
            excerpt = entry.get("source_excerpt")
            source_type = entry.get("source_type")
            source_id = str(entry.get("source_id") or "")
            source_ref = self._resolve_source(source_type, source_id, excerpt, message_by_id, issue, candidates, payload.get("tool_evidence", []))
            if not text or source_ref is None or self.store.is_tombstoned(session_id, source_ref.excerpt):
                continue
            if bare_resolution and source_ref.source_type == "user_message" and source_ref.source_id == current_message["id"]:
                # A generic resolution report cannot validate any particular previously suggested remedy.
                continue
            status = entry.get("status") if entry.get("status") in ("pending", "supported", "verified", "refuted") else "pending"
            if status in ("verified", "refuted"):
                markers = ("我确认", "我验证", "测试通过", "确实有效", "解决了", "已修复") if status == "verified" else ("无效", "没用", "失败", "没有解决", "不管用", "仍然", "排除")
                if source_ref.source_type != "user_message" or not any(marker in source_ref.excerpt for marker in markers):
                    status = "supported" if source_ref.source_type in ("candidate", "source_issue", "tool_evidence") else "pending"
            record = MemoryRecord(
                kind="experience", text=text,
                scope="repository", source_excerpt=source_ref.excerpt[:240],
                memory_id="mem_" + hashlib.sha256(f"{session_id}:{payload['message_id']}:{source_ref.source_type}:{source_ref.source_id}:{source_ref.excerpt}:{entry.get('entry_type')}".encode()).hexdigest()[:16],
                repository_id=repository_id, source_session_id=session_id,
                status=status, origin="user" if source_ref.source_type == "user_message" else "assistant",
                entry_type=entry.get("entry_type") if entry.get("entry_type") in (
                    "observation", "plan", "hypothesis", "attempt", "result", "summary"
                ) else "observation",
                verification_scope="solution" if entry.get("verification_scope") == "solution" and source_ref.source_type == "user_message" else "unspecified",
                case_id=case_id, source_refs=[source_ref], supporting_sessions=[session_id],
                modified_by="automatic",
            )
            target = entry.get("target_memory_id")
            if target:
                previous = next((item for item in payload.get("existing_records", []) if item.get("memory_id") == target), None)
                if previous:
                    self.store.revise_automatic(target, record, previous["revision"])
            elif case_id:
                self.store.add_record(record)

        preferences = extracted.get("preferences") if isinstance(extracted.get("preferences"), list) else []
        for preference in preferences[:4]:
            if not isinstance(preference, dict) or not isinstance(preference.get("text"), str):
                continue
            text = preference["text"].strip()[:300]
            message_id = str(preference.get("source_message_id") or "")
            source_message = message_by_id.get(message_id)
            excerpt = preference.get("source_excerpt")
            if not text or not source_message or not isinstance(excerpt, str) or excerpt not in source_message["content"]:
                continue
            if self.store.is_tombstoned(session_id, excerpt):
                continue
            long_term = any(marker in excerpt for marker in _LONG_TERM_MARKERS)
            explicit = bool(preference.get("explicit")) and long_term
            scope = preference.get("scope")
            if scope == "current" and not explicit:
                continue
            scope = "global" if scope == "global" and explicit else "repository"
            existing = next((record for record in self.store.list(repository_id)
                             if record.kind == "preference" and record.text.casefold() == text.casefold()), None)
            distinct_sessions = set(existing.supporting_sessions if existing else ())
            distinct_sessions.add(session_id)
            status = "active" if explicit or len(distinct_sessions) >= 2 else "observed"
            record = MemoryRecord(
                kind="preference", text=text, scope=scope, source_excerpt=excerpt[:240],
                memory_id=f"mem_{__import__('uuid').uuid4().hex[:16]}",
                repository_id=None if scope == "global" else repository_id,
                source_session_id=session_id, status=status,
                origin="explicit" if explicit else "inferred", entry_type="preference",
                source_refs=[MemorySourceRef(source_type="user_message", source_id=message_id,
                                             excerpt=excerpt[:240])],
                supporting_sessions=sorted(distinct_sessions), modified_by="automatic",
            )
            self.store.add_record(record)

    @staticmethod
    def _resolve_source(source_type: Any, source_id: str, excerpt: Any,
                        messages: dict[str, dict[str, Any]], issue: dict[str, Any],
                        candidates: list[dict[str, Any]], evidence: list[dict[str, Any]] | None = None) -> MemorySourceRef | None:
        if not isinstance(excerpt, str) or not excerpt.strip():
            return None
        excerpt = excerpt.strip()
        if source_type == "user_message" and source_id in messages:
            message = messages[source_id]
            return MemorySourceRef(source_type="user_message", source_id=source_id,
                                    excerpt=excerpt) if excerpt in message.get("content", "") else None
        if source_type == "source_issue":
            content = f"{issue.get('title', '')}\n{issue.get('body', '')}"
            if excerpt in content:
                return MemorySourceRef(source_type="source_issue", source_id=str(issue.get("url", "source_issue")),
                                       excerpt=excerpt, url=issue.get("url"))
        if source_type == "tool_evidence":
            item = next((item for item in evidence or [] if item.get("id") == source_id), None)
            if item and excerpt in item.get("text", ""):
                return MemorySourceRef(source_type="tool_evidence", source_id=source_id, excerpt=excerpt,
                                       url=item.get("url"))
        if source_type == "candidate":
            candidate = next((item for item in candidates if str(item.get("id")) == source_id), None)
            content = f"{candidate.get('title', '')}\n{candidate.get('body_snippet', '')}" if candidate else ""
            if candidate and excerpt in content:
                return MemorySourceRef(source_type="candidate", source_id=source_id, excerpt=excerpt,
                                       url=candidate.get("url"))
        return None

    @staticmethod
    def _user_confirmed_entry(source_excerpt: str, messages: Any) -> bool:
        confirmation_markers = ("我确认", "我验证", "测试通过", "确实有效", "解决了", "已修复")
        return any(marker in item.get("content", "") and source_excerpt in item.get("content", "")
                   for marker in confirmation_markers for item in messages)
