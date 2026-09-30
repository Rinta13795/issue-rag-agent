from src.chat.memory import MemoryStore
from src.chat.memory_service import MemoryService
from src.chat.models import MemoryCase, MemoryRecord


def _payload(session_id: str, message_id: str, case_id: str) -> dict:
    return {
        "session_id": session_id,
        "message_id": message_id,
        "repository_id": "owner_repo",
        "case_id": case_id,
        "investigation_summary": "",
        "user_messages": [{
            "id": message_id,
            "content": "我通常先查 PR。现在计划先核对关联 PR，还没验证。",
        }],
        "assistant_message": {"id": f"a-{message_id}", "content": "建议先看日志。"},
        "source_issue": None,
        "candidates": [],
    }


def test_memory_files_replay_revisions_and_deletion(tmp_path):
    root = tmp_path / "memory"
    store = MemoryStore(root)
    record = store.add_record(MemoryRecord(
        kind="preference", text="先查关联 PR", scope="global", source_excerpt="以后先查 PR",
        memory_id="pref-1", repository_id=None, source_session_id="session-1",
        status="active", origin="explicit", entry_type="preference",
        supporting_sessions=["session-1"],
    ))
    edited = store.update(record.memory_id, "先看关联 PR 的解决提交")

    restored = MemoryStore(root)
    loaded = restored.get(record.memory_id)
    assert loaded is not None
    assert loaded.text == "先看关联 PR 的解决提交"
    assert loaded.revision == 2
    assert loaded.revision_history[0].text == "先查关联 PR"
    assert (root / "MEMORY.md").exists()
    assert "先看关联 PR 的解决提交" in (root / "MEMORY.md").read_text(encoding="utf-8")
    assert restored.delete(record.memory_id)
    assert MemoryStore(root).get(record.memory_id) is None


def test_automatic_memory_records_pending_plans_and_promotes_repeated_preferences(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    case = MemoryCase(case_id="case-1", repository_id="owner_repo", title="登录失败", session_ids=["session-1"])
    store.create_or_update_case(case)

    def extract(payload):
        user = payload["user_messages"][0]
        return {
            "summary": "计划核对关联 PR；尚未验证。",
            "entries": [{
                "entry_type": "plan", "text": "核对关联 PR",
                "status": "pending", "source_type": "user_message",
                "source_id": user["id"], "source_excerpt": "计划先核对关联 PR",
            }],
            "preferences": [{
                "text": "调查时通常先查关联 PR", "scope": "repository",
                "explicit": False, "source_message_id": user["id"],
                "source_excerpt": "我通常先查 PR",
            }],
        }

    service = MemoryService(store, extract)
    payload = _payload("session-1", "message-1", case.case_id)
    first = service.enqueue(payload)
    assert first is not None
    assert service.enqueue(payload) == first
    service.wait_for_idle()
    assert store.job(first)["status"] == "completed"
    plan = next(item for item in store.list("owner_repo") if item.entry_type == "plan")
    assert plan.status == "pending"
    assert plan.source_refs[0].source_id == "message-1"
    pref = next(item for item in store.list("owner_repo") if item.kind == "preference")
    assert pref.status == "observed"

    second_payload = _payload("session-2", "message-2", case.case_id)
    second_payload["user_messages"][0]["content"] = "我通常先查 PR。"
    service.enqueue(second_payload)
    service.wait_for_idle()
    pref = next(item for item in store.list("owner_repo") if item.kind == "preference")
    assert pref.status == "active"
    assert set(pref.supporting_sessions) == {"session-1", "session-2"}


def test_organizer_rejects_assistant_only_preference_sources(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    service = MemoryService(store, lambda payload: {})
    payload = _payload("session-1", "message-1", "case-1")
    service._apply(payload, {
        "preferences": [{
            "text": "总是先看日志", "scope": "global", "explicit": True,
            "source_message_id": "assistant-message", "source_excerpt": "我总是先看日志",
        }],
    })
    assert store.list() == []


def test_failed_memory_job_can_retry_without_creating_duplicate(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    attempts = 0

    def extract(payload):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary model failure")
        return {}

    service = MemoryService(store, extract)
    payload = _payload("session-1", "message-1", "case-1")
    job_id = service.enqueue(payload)
    service.wait_for_idle()
    assert store.job(job_id)["status"] == "failed"
    assert service.enqueue(payload) == job_id
    service.wait_for_idle()
    assert attempts == 2
    assert store.job(job_id)["status"] == "completed"


def test_deleting_last_repository_experience_removes_markdown_projection(tmp_path):
    root = tmp_path / "memory"
    store = MemoryStore(root)
    record = MemoryRecord(
        kind="experience", text="缓存导致登录失败", scope="repository", source_excerpt="登录失败",
        memory_id="experience-1", repository_id="owner_repo", source_session_id="session-1",
        status="pending", origin="legacy", entry_type="observation",
    )
    store.add_record(record)
    projection = root / "repositories" / "owner_repo" / "MEMORY.md"
    assert projection.exists()
    store.delete(record.memory_id)
    assert not projection.exists()


def test_tool_pr_is_supported_not_user_verified_and_revision_survives_cleanup(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    store.create_or_update_case(MemoryCase(case_id="case-1", repository_id="owner_repo", title="日志问题"))
    service = MemoryService(store, lambda p: {})
    payload = _payload("s1", "m1", "case-1")
    payload["tool_evidence"] = [{"id":"pr:repo#2", "text":"This PR fixes INFO logging", "url":"https://github.com/a/b/pull/2"}]
    service._apply(payload, {"entries":[{"entry_type":"result", "text":"PR 描述修复日志", "status":"verified", "source_type":"tool_evidence", "source_id":"pr:repo#2", "source_excerpt":"fixes INFO logging"}]})
    record = store.list()[0]
    assert record.status == "supported"
    payload2 = _payload("s1", "m2", "case-1")
    payload2["user_messages"] = [{"id":"m2", "content":"我测试后没有解决，还是有 INFO 日志"}]
    payload2["existing_records"] = [record.model_dump()]
    update = {"entries":[{"target_memory_id":record.memory_id, "entry_type":"result", "text":"用户测试后无效，仍有日志", "status":"refuted", "source_type":"user_message", "source_id":"m2", "source_excerpt":"没有解决，还是有 INFO 日志"}]}
    service._apply(payload2, update)
    updated = store.get(record.memory_id)
    assert updated.status == "refuted" and updated.revision == 2
    restored = MemoryStore(tmp_path / "memory").get(record.memory_id)
    assert restored.revision_history[0].text == "PR 描述修复日志"
    assert restored.source_refs[0].excerpt == "fixes INFO logging"
    # User edits take precedence over queued jobs with old snapshots.
    store.update(record.memory_id, "我的修订")
    service._apply(payload2, update)
    assert store.get(record.memory_id).text == "我的修订"
    store.delete(record.memory_id)
    service._apply(payload2, update)
    assert store.get(record.memory_id) is None


def test_explicit_preference_without_case_and_one_off_instruction(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    service = MemoryService(store, lambda p: {})
    payload = _payload("s1", "m1", None)
    payload["user_messages"] = [{"id":"m1", "content":"以后所有项目都先查关联 PR。这次先看日志。"}]
    service._apply(payload, {"preferences":[
        {"text":"先查关联 PR", "scope":"global", "explicit":True, "source_message_id":"m1", "source_excerpt":"以后所有项目都先查关联 PR"},
        {"text":"先看日志", "scope":"current", "explicit":False, "source_message_id":"m1", "source_excerpt":"这次先看日志"},
    ]})
    assert len(store.list()) == 1
    context = service.read_context("different_repo", "解决问题")
    assert context["preferences"][0]["text"] == "先查关联 PR"


def test_deleted_multisource_memory_not_regenerated_after_restart(tmp_path):
    from src.chat.models import MemorySourceRef
    root = tmp_path / "memory"
    store = MemoryStore(root)
    record = MemoryRecord(kind="preference",text="查 PR",scope="repository",source_excerpt="通常查 PR",memory_id="m",repository_id="r",source_session_id="s1",supporting_sessions=["s1","s2"],source_refs=[MemorySourceRef(source_type="user_message",source_id="u2",excerpt="习惯查 PR")])
    store.add_record(record)
    store.delete("m")
    restored = MemoryStore(root)
    assert restored.is_tombstoned("s2", "习惯查 PR")


def test_pending_job_restarts_once(tmp_path):
    root = tmp_path / "memory"
    store = MemoryStore(root)
    job, _ = store.enqueue_job(_payload("s1", "m1", None))
    store.set_job_status(job, "processing")
    calls = []
    restored = MemoryStore(root)
    service = MemoryService(restored, lambda p: calls.append(p) or {})
    service.wait_for_idle()
    assert restored.job(job)["status"] == "completed" and len(calls) == 1
    service.enqueue(_payload("s1", "m1", None))
    service.wait_for_idle()
    assert len(calls) == 1


def test_unspecified_resolution_cannot_validate_guessed_pr(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    store.create_or_update_case(MemoryCase(case_id="case-1",repository_id="owner_repo",title="INFO",summary="两种建议：PR 补丁或调整日志级别"))
    old = store.add_record(MemoryRecord(kind="experience",text="PR 描述修复",scope="repository",source_excerpt="Fixes #352",memory_id="pr",repository_id="owner_repo",source_session_id="s1",status="supported",case_id="case-1",modified_by="automatic"))
    payload = _payload("s1", "m1", "case-1")
    payload['user_messages'] = [{"id":"m1","content":"OK我解决了"}]
    payload['existing_records'] = [old.model_dump()]
    service = MemoryService(store, lambda p: {})
    service._apply(payload, {"summary":"用户应用 PR 成功", "entries":[{"target_memory_id":"pr","text":"PR 在用户环境验证有效","entry_type":"result","status":"verified","source_type":"user_message","source_id":"m1","source_excerpt":"解决了","verification_scope":"solution"}]})
    assert store.get("pr").status == "supported"
    result = next(r for r in store.list() if r.memory_id != "pr")
    assert result.status == "verified" and result.verification_scope == "problem"
    assert result.source_refs[0].excerpt == "OK我解决了"
    assert "用户应用 PR 成功" not in store.get_case("case-1").summary
    service._apply(payload, {})
    assert len(store.list()) == 2
