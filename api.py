"""Issue RAG Agent HTTP 服务：提供 REST 接口、SSE 事件流与前端静态资源托管。

启动：
    uvicorn api:app --host 127.0.0.1 --port 8000 --reload

接口：
    GET  /health                  存活探针（不触发模型加载）
    POST /triage                  同步兼容接口（返回 duplicate/similar/new 判断）
    POST /api/runs                创建可观测运行任务（立即返回 run_id）
    GET  /api/runs/{run_id}       获取指定 run 的状态快照
    GET  /api/runs/{run_id}/events SSE 节点级实时事件流
    GET  /api/examples            获取预设演示样例
    GET  /api/system              获取系统元数据与限制说明
    GET  /api/evaluation/summary  获取真实评测指标汇总
"""

import asyncio
import json
import os
import uuid
import time
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from src.demo.eval_service import get_system_info, load_evaluation_summary
from src.demo.examples import get_preset_examples
from src.demo.models import (
    CreateRunRequest,
    CreateRunResponse,
    EvaluationSummary,
    ExampleItem,
    RunSnapshot,
    RunStatus,
    SystemInfo,
)
from src.demo.run_store import get_run_store
from src.demo.runner import get_observable_runner
from src.chat.models import AnswerQuestionRequest, ChatRepository, ChatSession, ChatSessionSummary, ConfirmMemoryRequest, CreateChatSessionRequest, MemoryCase, MemoryProposal, MemoryRecord, MemorySettings, MemorySourceRef, PreviewIssueRequest, PublishIssueDraftRequest, SendChatMessageRequest, SendChatMessageResponse, SourceIssue, SyncRepositoryRequest, SyncRepositoryStatus, UpdateIssueDraftRequest, UpdateMemoryRequest, UpdateMemorySettingsRequest, utc_now
from src.chat.github_sync import GitHubPublishError, create_github_issue, fetch_issue, get_sync_manager, repository_id
from src.chat.repositories import list_repositories, repository_exists
from src.chat.service import get_chat_runner
from src.chat.store import get_chat_store
from src.chat.memory import get_memory_store

app = FastAPI(
    title="Issue RAG Agent API",
    description="面向开源社区的重复 Issue 智能分诊系统：双粒度混合检索 + Cross-Encoder 精排 + LangGraph 决策工作流",
    version="2.0.0",
)

# 允许本地开发前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 原有兼容接口 ====================

class TriageRequest(BaseModel):
    issue_text: str = Field(min_length=1, max_length=20000, description="新 issue 原始文本")


class TriageResponse(BaseModel):
    decision: str
    confidence: float
    related_issues: list[str]
    reasoning: str
    retry_count: int


@app.get("/health")
def health() -> dict:
    """存活状态探针。"""
    return {"status": "ok"}


@app.post("/triage", response_model=TriageResponse)
def triage(request: TriageRequest) -> TriageResponse:
    """同步兼容入口：输入新 issue 文本，返回重复检测结果。"""
    from src.agent.graph import run_agent

    try:
        result = run_agent(request.issue_text.strip())
    except Exception as exc:
        logger.exception("triage 处理失败")
        raise HTTPException(status_code=500, detail=f"triage 失败：{type(exc).__name__}") from exc

    return TriageResponse(
        decision=result["decision"],
        confidence=result["confidence"],
        related_issues=[str(issue_id) for issue_id in result["related_issues"]],
        reasoning=result["reasoning"],
        retry_count=result["retry_count"],
    )


# ==================== 可观测 Demo 接口 ====================

@app.post("/api/runs", response_model=CreateRunResponse)
def create_run(request: CreateRunRequest) -> CreateRunResponse:
    """创建异步可观测运行任务，立即返回 run_id。"""
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    clean_text = request.issue_text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="issue_text 不能为空")

    store = get_run_store()
    snapshot = store.create_run(run_id, clean_text)

    runner = get_observable_runner()
    runner.submit_run(run_id, clean_text)

    return CreateRunResponse(
        run_id=run_id,
        status=snapshot.status,
        created_at=snapshot.created_at,
    )


@app.get("/api/runs/{run_id}", response_model=RunSnapshot)
def get_run_snapshot(run_id: str) -> RunSnapshot:
    """获取运行快照。"""
    store = get_run_store()
    snapshot = store.get_run(run_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail=f"未找到 run_id={run_id}")
    return snapshot


@app.get("/api/runs/{run_id}/events")
async def get_run_events(run_id: str):
    """Server-Sent Events (SSE) 事件流：推送节点生命周期事件。"""
    store = get_run_store()
    snapshot = store.get_run(run_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail=f"未找到 run_id={run_id}")

    queue: asyncio.Queue = asyncio.Queue()
    store.add_listener(run_id, queue)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # 首先发送一次当前初始状态
            init_data = json.dumps({"event": "run.snapshot", "data": snapshot.model_dump()})
            yield f"data: {init_data}\n\n"

            if snapshot.status in (RunStatus.COMPLETED, RunStatus.FAILED):
                return

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    msg = json.dumps(event)
                    yield f"data: {msg}\n\n"
                    if event.get("event") in ("run.completed", "run.failed"):
                        break
                except asyncio.TimeoutError:
                    # 发送保活心跳
                    yield ": ping\n\n"
                    current = store.get_run(run_id)
                    if current and current.status in (RunStatus.COMPLETED, RunStatus.FAILED):
                        break
        finally:
            store.remove_listener(run_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/examples", response_model=list[ExampleItem])
def get_examples() -> list[ExampleItem]:
    """获取预设演示样例列表。"""
    return get_preset_examples()


@app.get("/api/system", response_model=SystemInfo)
def get_system() -> SystemInfo:
    """获取系统架构与模型状态。"""
    return get_system_info()


@app.get("/api/evaluation/summary", response_model=EvaluationSummary)
def get_evaluation() -> EvaluationSummary:
    """获取真实评测指标汇总。"""
    return load_evaluation_summary()


# ==================== 本地多轮对话 ====================

@app.get("/api/chat/repositories", response_model=list[ChatRepository])
def get_chat_repositories() -> list[ChatRepository]:
    return list_repositories()


@app.post("/api/chat/issues/preview", response_model=SourceIssue)
def preview_chat_issue(request: PreviewIssueRequest) -> SourceIssue:
    """实时读取用户指定的公开 Issue，供仓库同步与导入预览。"""
    try:
        return fetch_issue(request.issue_url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/chat/repositories/sync", response_model=SyncRepositoryStatus, status_code=202)
def sync_chat_repository(request: SyncRepositoryRequest) -> SyncRepositoryStatus:
    try:
        return get_sync_manager().start(request.repository)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/chat/repositories/sync/{job_id}", response_model=SyncRepositoryStatus)
def get_chat_repository_sync(job_id: str) -> SyncRepositoryStatus:
    job = get_sync_manager().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="同步任务不存在或服务已重启")
    return job


@app.get("/api/chat/sessions", response_model=list[ChatSessionSummary])
def list_chat_sessions() -> list[ChatSessionSummary]:
    return get_chat_store().list_sessions()


@app.post("/api/chat/sessions", response_model=ChatSession)
def create_chat_session(request: CreateChatSessionRequest) -> ChatSession:
    if not repository_exists(request.repository_id):
        raise HTTPException(status_code=422, detail="仓库未建立本地索引，请先同步后再选择")
    source_issue = None
    if request.issue_url:
        try:
            source_issue = fetch_issue(request.issue_url)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if repository_id(source_issue.repository) != request.repository_id and not (
            request.repository_id == "openharness" and source_issue.repository.lower() == "hkuds/openharness"
        ):
            raise HTTPException(status_code=422, detail="Issue 不属于当前选择的仓库")
    memory_case_id = None
    investigation_summary = ""
    if request.case_id:
        case = get_memory_store().get_case(request.case_id)
        if case is None or case.repository_id != request.repository_id:
            raise HTTPException(status_code=422, detail="问题案例不存在或不属于当前仓库")
        if source_issue and case.source_issue_id and case.source_issue_id != f"{source_issue.repository}#{source_issue.number}":
            raise HTTPException(status_code=422, detail="当前 Issue 与所选问题案例不匹配")
        memory_case_id, investigation_summary = case.case_id, case.summary
    session = get_chat_store().create(
        repository_id=request.repository_id, source_issue=source_issue,
        memory_case_id=memory_case_id, investigation_summary=investigation_summary,
    )
    if memory_case_id:
        case = get_memory_store().get_case(memory_case_id)
        if case:
            get_memory_store().create_or_update_case(case.model_copy(update={
                "session_ids": list(dict.fromkeys(case.session_ids + [session.session_id])),
                "updated_at": utc_now(),
            }))
    return session


@app.get("/api/chat/sessions/{session_id}", response_model=ChatSession)
def get_chat_session(session_id: str) -> ChatSession:
    session = get_chat_store().get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    return session


@app.get("/api/chat/sessions/{session_id}/events")
async def chat_session_events(session_id: str, request: Request):
    store = get_chat_store()
    if store.get(session_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    async def events():
        version, committed = -1, -1
        heartbeat = time.monotonic()
        while not await request.is_disconnected():
            snapshot, next_version, next_committed = store.stream_snapshot(session_id, version)
            if snapshot is not None:
                if next_committed != committed:
                    payload = snapshot.model_dump(exclude={"runtime_messages", "runtime_memory_snapshot"})
                    event = "snapshot"
                else:
                    payload = {"answer": snapshot.streaming_answer, "turn_id": snapshot.streaming_turn_id}
                    event = "answer"
                yield f"id: {next_version}\nevent: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                version, committed = next_version, next_committed
            if time.monotonic() - heartbeat >= 15:
                if store.get(session_id) is None:
                    yield 'event: expired\ndata: {}\n\n'
                    return
                yield ': heartbeat\n\n'
                heartbeat = time.monotonic()
            await asyncio.sleep(0.08)

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no",
    })


@app.post("/api/chat/sessions/{session_id}/messages", response_model=SendChatMessageResponse, status_code=202)
def send_chat_message(session_id: str, request: SendChatMessageRequest) -> SendChatMessageResponse:
    content = request.content.strip()
    if not content:
        raise HTTPException(status_code=422, detail="消息不能为空")
    store = get_chat_store()
    try:
        message_id, created = store.add_user_message(session_id, content, request.client_message_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="会话不存在或已过期") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if created:
        get_chat_runner().submit(session_id, message_id)
    session = store.get(session_id)
    return SendChatMessageResponse(message_id=message_id, session_id=session_id, status=session.status)


@app.post("/api/chat/sessions/{session_id}/answers", response_model=SendChatMessageResponse, status_code=202)
def answer_chat_question(session_id: str, request: AnswerQuestionRequest) -> SendChatMessageResponse:
    store = get_chat_store()
    try:
        message_id, created = store.answer_question(session_id, request.question_id, request.answer,
                                                  request.client_message_id, request.cancelled)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="会话不存在或已过期") from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if created:
        get_chat_runner().submit(session_id, message_id)
    return SendChatMessageResponse(message_id=message_id, session_id=session_id, status=store.get(session_id).status)


@app.post("/api/chat/sessions/{session_id}/retry", response_model=SendChatMessageResponse, status_code=202)
def retry_chat_message(session_id: str) -> SendChatMessageResponse:
    store = get_chat_store()
    try:
        message_id = store.retry_failed_turn(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="会话不存在或已过期") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    get_chat_runner().submit(session_id, message_id)
    return SendChatMessageResponse(message_id=message_id, session_id=session_id, status="thinking")


@app.post("/api/chat/sessions/{session_id}/issue-draft", response_model=ChatSession)
def create_chat_issue_draft(session_id: str) -> ChatSession:
    """显式生成可编辑草稿，不执行 GitHub 写入。"""
    try:
        return get_chat_runner().service.create_draft(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="会话不存在或已过期") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("Issue 草稿生成失败 error_type={}", type(exc).__name__)
        raise HTTPException(status_code=502, detail="草稿生成失败；原对话没有丢失，请稍后重试") from exc


@app.patch("/api/chat/sessions/{session_id}/issue-draft", response_model=ChatSession)
def edit_chat_issue_draft(session_id: str, request: UpdateIssueDraftRequest) -> ChatSession:
    try:
        return get_chat_store().edit_issue_draft(session_id, request.title, request.body, request.version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="会话不存在或已过期") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/chat/sessions/{session_id}/issue-draft/publish", response_model=ChatSession)
def publish_chat_issue_draft(session_id: str, request: PublishIssueDraftRequest) -> ChatSession:
    """只发布用户确认的当前草稿版本；占用发布状态防止重复点击。"""
    if not request.confirmed:
        raise HTTPException(status_code=422, detail="发布前须确认公开仓库和草稿正文")
    store = get_chat_store()
    snapshot = store.get(session_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    repo = next((item for item in list_repositories() if item.id == snapshot.repository_id), None)
    if repo is None or not repo.github_url:
        raise HTTPException(status_code=422, detail="该对话没有可发布的 GitHub 仓库；可以复制草稿自行提交")
    try:
        claimed = store.claim_issue_publish(session_id, request.draft_id, request.version)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    draft = claimed.issue_draft
    if draft.status == "published":
        return claimed
    try:
        url, number = create_github_issue(repo.github_url.removeprefix("https://github.com/"), draft.title, draft.body)
    except GitHubPublishError as exc:
        store.finish_issue_publish(session_id, draft.draft_id, None, None, uncertain=not exc.definite)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return store.finish_issue_publish(session_id, draft.draft_id, url, number)


@app.post("/api/chat/sessions/{session_id}/memory-proposal", response_model=ChatSession)
def propose_chat_memory(session_id: str) -> ChatSession:
    """按需提炼一条待确认经验，不直接写长期记忆。"""
    try:
        return get_chat_runner().service.create_memory_proposal(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="会话不存在或已过期") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("记忆提炼失败 error_type={}", type(exc).__name__)
        raise HTTPException(status_code=502, detail="记忆提炼失败；对话没有丢失") from exc


@app.post("/api/chat/sessions/{session_id}/memory", response_model=MemoryRecord)
def confirm_chat_memory(session_id: str, request: ConfirmMemoryRequest) -> MemoryRecord:
    if not request.confirmed:
        raise HTTPException(status_code=422, detail="请先确认这条经验确实值得长期保存")
    store = get_chat_store()
    if request.kind == "experience" and request.scope != "repository":
        raise HTTPException(status_code=422, detail="处理经验必须绑定当前仓库")
    snapshot = store.get(session_id)
    try:
        original, repository_id = store.take_memory_proposal(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="会话不存在或已过期") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    proposal = MemoryProposal(kind=request.kind, scope=request.scope, text=request.text.strip(),
                              source_excerpt=original.source_excerpt)
    source_ref = None
    if snapshot:
        message = next((item for item in snapshot.messages
                        if item.role == "user" and original.source_excerpt in item.content), None)
        if message:
            source_ref = MemorySourceRef(source_type="user_message", source_id=message.id,
                                         excerpt=original.source_excerpt)
        elif snapshot.source_issue and original.source_excerpt in (
            snapshot.source_issue.title + "\n" + snapshot.source_issue.body
        ):
            source_ref = MemorySourceRef(source_type="source_issue", source_id=snapshot.source_issue.url,
                                         excerpt=original.source_excerpt, url=snapshot.source_issue.url)
    try:
        return get_memory_store().add(
            proposal, repository_id, session_id, source_ref=source_ref,
            case_id=snapshot.memory_case_id if snapshot else None,
        )
    except Exception:
        store.restore_memory_proposal(session_id, original)
        raise


@app.get("/api/chat/memories", response_model=list[MemoryRecord])
def list_chat_memories(repository_id: str | None = None) -> list[MemoryRecord]:
    return get_memory_store().list(repository_id)


@app.patch("/api/chat/memories/{memory_id}", response_model=MemoryRecord)
def update_chat_memory(memory_id: str, request: UpdateMemoryRequest) -> MemoryRecord:
    try:
        return get_memory_store().update(memory_id, request.text, request.status)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="记忆不存在") from exc


@app.get("/api/chat/memory-settings", response_model=MemorySettings)
def get_chat_memory_settings() -> MemorySettings:
    return MemorySettings(auto_capture=get_memory_store().auto_capture)


@app.patch("/api/chat/memory-settings", response_model=MemorySettings)
def update_chat_memory_settings(request: UpdateMemorySettingsRequest) -> MemorySettings:
    get_memory_store().set_auto_capture(request.auto_capture)
    return MemorySettings(auto_capture=request.auto_capture)


@app.get("/api/chat/memory-cases", response_model=list[MemoryCase])
def list_chat_memory_cases(repository_id: str) -> list[MemoryCase]:
    return get_memory_store().list_cases(repository_id)


@app.get("/api/chat/memory-cases/{case_id}")
def get_chat_memory_case(case_id: str) -> dict:
    memory_store = get_memory_store()
    case = memory_store.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="问题案例不存在")
    return {
        "case": case.model_dump(mode="json"),
        "entries": [record.model_dump(mode="json") for record in memory_store.list(case.repository_id)
                    if record.case_id == case_id],
    }


@app.delete("/api/chat/memories/{memory_id}")
def delete_chat_memory(memory_id: str) -> dict:
    if not get_memory_store().delete(memory_id):
        raise HTTPException(status_code=404, detail="记忆不存在")
    return {"deleted": True}


# ==================== 前端静态资源挂载 ====================

frontend_dist = Path(__file__).parent / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
