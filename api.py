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
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
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


# ==================== 前端静态资源挂载 ====================

frontend_dist = Path(__file__).parent / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
