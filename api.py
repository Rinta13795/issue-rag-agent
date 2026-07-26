"""Issue RAG Agent HTTP 服务：把 run_agent 暴露为 REST 接口。

启动：
    uvicorn api:app --host 127.0.0.1 --port 8000

接口：
    GET  /health   存活探针（不触发模型加载）
    POST /triage   请求体 {"issue_text": "..."}，返回 duplicate/similar/new 判断

模型与索引在首个 /triage 请求时惰性加载；生产部署可在启动后先打一次预热请求。
"""

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

app = FastAPI(
    title="Issue RAG Agent",
    description="Issue 重复检测：输入新 issue 文本，输出 duplicate/similar/new 判断",
    version="2.0.0",
)


class TriageRequest(BaseModel):
    """POST /triage 请求体：只需要新 issue 的原始文本。"""

    issue_text: str = Field(min_length=1, max_length=20000, description="新 issue 原始文本")


class TriageResponse(BaseModel):
    """POST /triage 响应体：与 run_agent 返回字段一一对应。"""

    decision: str
    confidence: float
    related_issues: list[str]
    reasoning: str
    retry_count: int


@app.get("/health")
def health() -> dict:
    """输入无，输出存活状态；不触发模型加载，可用于容器探针。"""
    return {"status": "ok"}


@app.post("/triage", response_model=TriageResponse)
def triage(request: TriageRequest) -> TriageResponse:
    """输入新 issue 文本，输出重复检测判断结果。"""
    # 延迟导入：服务启动和 /health 探测不需要加载 Embedding/Reranker/索引。
    from src.agent.graph import run_agent

    try:
        result = run_agent(request.issue_text.strip())
    except Exception as exc:  # noqa: BLE001 —— 对外接口统一转成 500，避免泄漏内部堆栈。
        logger.exception("triage 处理失败")
        raise HTTPException(status_code=500, detail=f"triage 失败：{type(exc).__name__}") from exc

    return TriageResponse(
        decision=result["decision"],
        confidence=result["confidence"],
        related_issues=[str(issue_id) for issue_id in result["related_issues"]],
        reasoning=result["reasoning"],
        retry_count=result["retry_count"],
    )
