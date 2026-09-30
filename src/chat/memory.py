"""File-backed long-term memory, modeled after workspace memory files."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.chat.models import (
    MemoryCase,
    MemoryProposal,
    MemoryRecord,
    MemoryRevision,
    MemorySourceRef,
)


def _bigrams(value: str) -> set[str]:
    text = re.sub(r"\s+", " ", value.casefold()).strip()
    return {text[index:index + 2] for index in range(max(len(text) - 1, 0))}


class MemoryStore:
    """Append-only JSONL source of truth with readable Markdown memory files.

    ``db_path`` is kept as a compatibility argument for callers and tests, but
    points to a JSONL event file. No SQLite connection is opened here.
    """

    def __init__(self, db_path: str | Path | None = None):
        self._lock = threading.RLock()
        self.root = Path(db_path) if db_path is not None else None
        if self.root is not None and self.root.suffix:
            self.root.parent.mkdir(parents=True, exist_ok=True)
            self.events_path = self.root
            self.files_root = self.root.parent / f"{self.root.stem}_files"
            self.jobs_path = self.root.parent / f"{self.root.stem}_jobs.jsonl"
        elif self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            self.events_path = self.root / "events.jsonl"
            self.files_root = self.root
            self.jobs_path = self.root / "jobs.jsonl"
        else:
            self.events_path = None
            self.files_root = None
            self.jobs_path = None
        self._records: dict[str, MemoryRecord] = {}
        self._cases: dict[str, MemoryCase] = {}
        self._jobs: dict[str, dict[str, Any]] = {}
        self._job_keys: dict[str, str] = {}
        self._tombstones: set[str] = set()
        self._auto_capture = self.events_path is not None
        self._restore()

    def _append(self, path: Path | None, event: dict[str, Any]) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()

    @staticmethod
    def _read_events(path: Path | None) -> list[dict[str, Any]]:
        if path is None or not path.exists():
            return []
        events = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
        return events

    def _restore(self) -> None:
        for event in self._read_events(self.events_path):
            event_type = event.get("type")
            if event_type in ("record_added", "record_updated"):
                try:
                    record = MemoryRecord.model_validate(event["record"])
                except (KeyError, ValueError):
                    continue
                self._records[record.memory_id] = record
            elif event_type == "record_deleted":
                memory_id = event.get("memory_id")
                if isinstance(memory_id, str):
                    removed = self._records.pop(memory_id, None)
                    self._tombstones.update(event.get("source_keys", []))
                    source_key = event.get("source_key")
                    if isinstance(source_key, str):
                        self._tombstones.add(source_key)
                    elif removed:
                        self._tombstones.add(self.source_key(removed.source_session_id, removed.source_excerpt))
            elif event_type == "case_updated":
                try:
                    case = MemoryCase.model_validate(event["case"])
                except (KeyError, ValueError):
                    continue
                self._cases[case.case_id] = case
            elif event_type == "auto_capture":
                self._auto_capture = bool(event.get("enabled", True))
        for event in self._read_events(self.jobs_path):
            job_id = event.get("job_id")
            if not isinstance(job_id, str):
                continue
            if event.get("type") == "job_enqueued":
                job = dict(event.get("job") or {})
                job.update(job_id=job_id, status="queued")
                self._jobs[job_id] = job
                self._job_keys[job.get("turn_key", "")] = job_id
            elif event.get("type") == "job_extracted" and job_id in self._jobs:
                self._jobs[job_id]["extracted"] = event.get("extracted", {})
            elif event.get("type") == "job_status" and job_id in self._jobs:
                self._jobs[job_id]["status"] = event.get("status", "queued")
                self._jobs[job_id]["error"] = event.get("error")
        for job in self._jobs.values():
            if job.get("status") == "processing":
                job["status"] = "queued"

    @staticmethod
    def source_key(session_id: str, excerpt: str) -> str:
        normalized = re.sub(r"\s+", " ", excerpt).strip().casefold()
        return f"{session_id}:{normalized}"

    def is_tombstoned(self, session_id: str, excerpt: str) -> bool:
        return self.source_key(session_id, excerpt) in self._tombstones

    def add(
        self,
        proposal: MemoryProposal,
        repository_id: str | None,
        session_id: str,
        source_ref: MemorySourceRef | None = None,
        case_id: str | None = None,
    ) -> MemoryRecord:
        scope_id = repository_id if proposal.scope == "repository" else None
        if proposal.scope == "repository" and not scope_id:
            raise ValueError("仓库记忆缺少仓库归属")
        record = MemoryRecord(
            **proposal.model_dump(), memory_id=f"mem_{uuid.uuid4().hex[:16]}",
            repository_id=scope_id, source_session_id=session_id,
            status="verified" if proposal.kind == "experience" else "active",
            origin="user", entry_type="preference" if proposal.kind == "preference" else "result",
            supporting_sessions=[session_id],
            source_refs=[source_ref] if source_ref else [], case_id=case_id,
        )
        return self.add_record(record)

    def add_record(self, record: MemoryRecord) -> MemoryRecord:
        with self._lock:
            if record.memory_id in self._records:
                return self._records[record.memory_id]
            source_key = self.source_key(record.source_session_id, record.source_excerpt)
            if source_key in self._tombstones:
                return record
            existing = next((item for item in self._records.values()
                             if item.kind == record.kind and item.case_id == record.case_id
                             and item.repository_id == record.repository_id
                             and item.text.casefold() == record.text.casefold()
                             and item.status != "refuted"), None)
            if existing:
                sessions = list(dict.fromkeys(existing.supporting_sessions + record.supporting_sessions))
                refs = {(ref.source_type, ref.source_id, ref.excerpt): ref for ref in existing.source_refs}
                refs.update({(ref.source_type, ref.source_id, ref.excerpt): ref for ref in record.source_refs})
                next_status = existing.status
                if (existing.kind == "preference" and existing.origin == "inferred"
                        and existing.modified_by != "user" and len(sessions) >= 2):
                    next_status = "active"
                revisions = existing.revision_history
                revision = existing.revision
                if existing.modified_by != "user" and record.status in ("supported", "verified", "refuted"):
                    status_rank = {"observed": 0, "pending": 0, "supported": 1, "verified": 2, "refuted": 2, "active": 1}
                    if status_rank.get(record.status, 0) >= status_rank.get(next_status, 0):
                        next_status = record.status
                if next_status != existing.status:
                    revisions = revisions + [MemoryRevision(
                        revision=revision, text=existing.text, status=existing.status,
                        changed_by="automatic", reason="调查中出现新的支持或验证证据",
                    )]
                    revision += 1
                updated = existing.model_copy(update={
                    "supporting_sessions": sessions, "status": next_status,
                    "source_refs": list(refs.values()), "revision_history": revisions, "revision": revision,
                })
                self._records[existing.memory_id] = updated
                self._append(self.events_path, {"type": "record_updated", "record": updated.model_dump(mode="json")})
                self._render_markdown()
                return updated
            self._records[record.memory_id] = record
            self._append(self.events_path, {"type": "record_added", "record": record.model_dump(mode="json")})
            self._render_markdown()
            return record

    def revise_automatic(self, memory_id: str, record: MemoryRecord, expected_revision: int) -> bool:
        """后台仅修订快照中未被手动编辑的记录，删除后不会重建。"""
        with self._lock:
            current = self._records.get(memory_id)
            if (current is None or current.modified_by == "user" or current.revision != expected_revision
                    or current.repository_id != record.repository_id or current.case_id != record.case_id
                    or current.kind != record.kind):
                return False
            if self.is_tombstoned(record.source_session_id, record.source_excerpt):
                return False
            refs = {(ref.source_type, ref.source_id, ref.excerpt): ref for ref in current.source_refs}
            refs.update({(ref.source_type, ref.source_id, ref.excerpt): ref for ref in record.source_refs})
            if current.text == record.text and current.status == record.status and len(refs) == len(current.source_refs):
                return True
            changed = current.model_copy(update={
                "text": record.text, "status": record.status, "verification_scope": record.verification_scope, "source_refs": list(refs.values()),
                "supporting_sessions": list(dict.fromkeys(current.supporting_sessions + record.supporting_sessions)),
                "revision": current.revision + 1,
                "revision_history": current.revision_history + [MemoryRevision(revision=current.revision,
                    text=current.text, status=current.status, changed_by="automatic", reason="新增调查证据修订")],
            })
            self._records[memory_id] = changed
            self._append(self.events_path, {"type": "record_updated", "record": changed.model_dump(mode="json")})
            self._render_markdown()
            return True

    def list(self, repository_id: str | None = None) -> list[MemoryRecord]:
        with self._lock:
            records = list(self._records.values())
        if repository_id:
            records = [item for item in records if item.repository_id in (None, repository_id)]
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    def get(self, memory_id: str) -> MemoryRecord | None:
        return self._records.get(memory_id)

    def update(self, memory_id: str, text: str, status: str | None = None, reason: str = "用户编辑") -> MemoryRecord:
        with self._lock:
            current = self._records.get(memory_id)
            if current is None:
                raise KeyError(memory_id)
            next_status = status or current.status
            revision = MemoryRevision(revision=current.revision, text=current.text, status=current.status,
                                      changed_by="user", reason=reason)
            changed = current.model_copy(update={
                "text": text.strip(), "status": next_status, "revision": current.revision + 1,
                "revision_history": current.revision_history + [revision], "modified_by": "user",
            })
            self._records[memory_id] = changed
            self._append(self.events_path, {"type": "record_updated", "record": changed.model_dump(mode="json")})
            self._render_markdown()
            return changed

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            record = self._records.pop(memory_id, None)
            if record is None:
                return False
            source_key = self.source_key(record.source_session_id, record.source_excerpt)
            self._tombstones.add(source_key)
            for session_id in record.supporting_sessions:
                self._tombstones.add(self.source_key(session_id, record.source_excerpt))
            for source_ref in record.source_refs:
                if source_ref.source_type == "user_message":
                    for session_id in record.supporting_sessions:
                        self._tombstones.add(self.source_key(session_id, source_ref.excerpt))
            self._append(self.events_path, {"type": "record_deleted", "memory_id": memory_id, "source_key": source_key, "source_keys": sorted(self._tombstones)})
            self._render_markdown()
            return True

    def select(self, repository_id: str | None, query: str, limit: int = 6) -> list[MemoryRecord]:
        records = self.list(repository_id)
        preferences = [item for item in records if item.kind == "preference" and item.status == "active"][:3]
        query_grams = _bigrams(query)
        cases = [item for item in records if item.kind == "experience"
                 and item.repository_id == repository_id and item.status != "refuted"]
        scored = sorted(((len(_bigrams(item.text) & query_grams), item) for item in cases),
                        key=lambda pair: pair[0], reverse=True)
        chosen = preferences + [item for score, item in scored if score > 0][:3]
        return chosen[:limit]

    def create_or_update_case(self, case: MemoryCase) -> MemoryCase:
        with self._lock:
            current = self._cases.get(case.case_id)
            if current:
                case = case.model_copy(update={
                    "session_ids": list(dict.fromkeys(current.session_ids + case.session_ids)),
                    "source_issue_id": case.source_issue_id or current.source_issue_id,
                    "source_issue_url": case.source_issue_url or current.source_issue_url,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
            self._cases[case.case_id] = case
            self._append(self.events_path, {"type": "case_updated", "case": case.model_dump(mode="json")})
            self._render_markdown()
            return case

    def get_case(self, case_id: str) -> MemoryCase | None:
        return self._cases.get(case_id)

    def find_case(self, repository_id: str, source_issue_id: str | None, session_id: str) -> MemoryCase | None:
        if source_issue_id:
            match = next((case for case in self._cases.values()
                          if case.repository_id == repository_id and case.source_issue_id == source_issue_id), None)
            if match:
                return match
        return next((case for case in self._cases.values()
                     if case.repository_id == repository_id and session_id in case.session_ids), None)

    def list_cases(self, repository_id: str) -> list[MemoryCase]:
        return sorted((case for case in self._cases.values() if case.repository_id == repository_id),
                      key=lambda case: case.updated_at, reverse=True)

    def select_cases(self, repository_id: str, query: str, limit: int = 3) -> list[MemoryCase]:
        cases = self.list_cases(repository_id)
        query_grams = _bigrams(query)
        records_by_case: dict[str, list[str]] = {}
        for record in self._records.values():
            if record.kind == "experience" and record.case_id:
                records_by_case.setdefault(record.case_id, []).append(record.text)
        scored = []
        for case in cases:
            searchable = " ".join([case.title, case.summary, *records_by_case.get(case.case_id, [])])
            score = len(_bigrams(searchable) & query_grams)
            if score:
                scored.append((score, case))
        scored.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        return [case for _, case in scored[:limit]]

    def update_case_summary(self, case_id: str, summary: str, session_id: str) -> MemoryCase | None:
        with self._lock:
            current = self._cases.get(case_id)
            if current is None:
                return None
            changed = current.model_copy(update={
                "summary": summary[:1600],
                "session_ids": list(dict.fromkeys(current.session_ids + [session_id])),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            self._cases[case_id] = changed
            self._append(self.events_path, {"type": "case_updated", "case": changed.model_dump(mode="json")})
            self._render_markdown()
            return changed

    def set_auto_capture(self, enabled: bool) -> None:
        self._auto_capture = enabled
        with self._lock:
            self._append(self.events_path, {"type": "auto_capture", "enabled": enabled})

    @property
    def auto_capture(self) -> bool:
        return self._auto_capture

    def enqueue_job(self, payload: dict[str, Any]) -> tuple[str, bool]:
        with self._lock:
            turn_key = f"{payload['session_id']}:{payload['message_id']}"
            existing_id = self._job_keys.get(turn_key)
            if existing_id:
                existing = self._jobs[existing_id]
                if existing.get("status") == "failed":
                    existing.update(status="queued", error=None)
                    self._append(self.jobs_path, {
                        "type": "job_status", "job_id": existing_id, "status": "queued", "error": None,
                    })
                    return existing_id, True
                return existing_id, False
            job_id = f"memjob_{uuid.uuid4().hex[:16]}"
            job = {"turn_key": turn_key, "payload": payload, "status": "queued", "error": None}
            self._jobs[job_id] = {"job_id": job_id, **job}
            self._job_keys[turn_key] = job_id
            self._append(self.jobs_path, {"type": "job_enqueued", "job_id": job_id, "job": job})
            return job_id, True

    def save_job_extraction(self, job_id: str, extracted: dict[str, Any]) -> None:
        with self._lock:
            self._append(self.jobs_path, {"type": "job_extracted", "job_id": job_id, "extracted": extracted})
            self._jobs[job_id]["extracted"] = extracted

    def set_job_status(self, job_id: str, status: str, error: str | None = None) -> None:
        with self._lock:
            if job_id not in self._jobs:
                return
            self._jobs[job_id].update(status=status, error=error)
            self._append(self.jobs_path, {"type": "job_status", "job_id": job_id, "status": status, "error": error})

    def pending_jobs(self) -> list[dict[str, Any]]:
        return [dict(job) for job in self._jobs.values() if job.get("status") in ("queued", "failed")]

    def job(self, job_id: str) -> dict[str, Any] | None:
        job = self._jobs.get(job_id)
        return dict(job) if job else None

    def cases_for_session(self, session_id: str) -> list[MemoryCase]:
        return [case for case in self._cases.values() if session_id in case.session_ids]

    def _render_markdown(self) -> None:
        if self.files_root is None:
            return
        self.files_root.mkdir(parents=True, exist_ok=True)
        preferences = [item for item in self.list() if item.kind == "preference"]
        lines = ["# Long-term preferences", "", "Automatically maintained local memory.", ""]
        if not preferences:
            lines.append("No preferences recorded yet.")
        for item in preferences:
            lines.extend([f"## {item.text}", f"- Status: {item.status} · Origin: {item.origin} · Scope: {item.scope}",
                          f"- Source: {item.source_excerpt}", f"- Updated: {item.created_at}", ""])
        temp = self.files_root / ".preferences.md.tmp"
        temp.write_text("\n".join(lines), encoding="utf-8")
        temp.replace(self.files_root / "MEMORY.md")
        repository_dir = self.files_root / "repositories"
        standalone = [item for item in self._records.values()
                      if item.kind == "experience" and item.case_id is None and item.repository_id]
        repository_ids = {item.repository_id for item in standalone}
        safe_names = {
            re.sub(r"[^a-zA-Z0-9._-]+", "_", repository_id or "unknown")[:100]
            for repository_id in repository_ids
        }
        if repository_dir.exists():
            for stale_path in repository_dir.glob("*/MEMORY.md"):
                if stale_path.parent.name not in safe_names:
                    stale_path.unlink(missing_ok=True)
        for repository_id in repository_ids:
            safe_repository = re.sub(r"[^a-zA-Z0-9._-]+", "_", repository_id or "unknown")[:100]
            repo_path = repository_dir / safe_repository / "MEMORY.md"
            repo_path.parent.mkdir(parents=True, exist_ok=True)
            repo_lines = [f"# Experiences · {repository_id}", ""]
            for item in standalone:
                if item.repository_id != repository_id:
                    continue
                repo_lines.extend([
                    f"## {item.entry_type}: {item.status}", item.text,
                    f"Source: {item.source_excerpt}", f"Updated: {item.created_at}", "",
                ])
            tmp = repo_path.with_suffix(".md.tmp")
            tmp.write_text("\n".join(repo_lines), encoding="utf-8")
            tmp.replace(repo_path)
        case_dir = self.files_root / "cases"
        case_dir.mkdir(exist_ok=True)
        for case in self._cases.values():
            entries = [record for record in self._records.values() if record.case_id == case.case_id]
            body = [f"# {case.title}", "", case.summary or "No investigation summary yet.", ""]
            for record in sorted(entries, key=lambda item: item.created_at):
                body.extend([f"## {record.entry_type}: {record.status}", record.text,
                             f"Source: {record.source_excerpt}", ""])
            path = case_dir / f"{case.case_id}.md"
            tmp = path.with_suffix(".md.tmp")
            tmp.write_text("\n".join(body), encoding="utf-8")
            tmp.replace(path)


_MEMORY_STORE: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    global _MEMORY_STORE
    if _MEMORY_STORE is None:
        root = Path(__file__).resolve().parents[2] / ".local" / "issue-rag-memory"
        _MEMORY_STORE = MemoryStore(root)
        _migrate_legacy_sqlite(_MEMORY_STORE, root.parent / "chat_memories.sqlite3")
    return _MEMORY_STORE


def _migrate_legacy_sqlite(store: MemoryStore, legacy_path: Path) -> None:
    """Read old confirmed records once and preserve them in the file journal."""
    marker = store.files_root / ".legacy-imported" if store.files_root else None
    if marker is None or marker.exists() or not legacy_path.exists():
        return
    try:
        connection = sqlite3.connect(f"file:{legacy_path}?mode=ro", uri=True)
        rows = connection.execute("SELECT payload FROM chat_memories").fetchall()
        connection.close()
    except sqlite3.Error:
        return
    for (payload,) in rows:
        try:
            legacy = MemoryRecord.model_validate_json(payload)
            record = legacy.model_copy(update={
                "status": "active" if legacy.kind == "preference" else "pending",
                "origin": "legacy", "modified_by": "user",
            })
            store.add_record(record)
        except (ValueError, TypeError):
            continue
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("Imported legacy SQLite memories.\n", encoding="utf-8")
