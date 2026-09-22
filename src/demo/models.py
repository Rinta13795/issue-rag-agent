"""Demo 可观测运行适配层：Pydantic 数据模型。"""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class NodeStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DecisionType(str, Enum):
    DUPLICATE = "duplicate"
    SIMILAR = "similar"
    NEW = "new"


class QueryAnalysisInfo(BaseModel):
    rewritten_query: str = ""
    keywords: list[str] = Field(default_factory=list)
    component: Optional[str] = None
    component_filter_applied: bool = False
    component_filter_note: Optional[str] = None


class CandidateInfo(BaseModel):
    id: str
    title: str = ""
    body_snippet: str = ""
    score: Optional[float] = None
    rerank_score: Optional[float] = None
    is_related: bool = False


class DecisionInfo(BaseModel):
    decision: str
    confidence: float
    related_issues: list[str] = Field(default_factory=list)
    reasoning: str = ""
    retry_count: int = 1


class RetryRoundInfo(BaseModel):
    round: int
    rewritten_query: str = ""
    keywords: list[str] = Field(default_factory=list)
    component: Optional[str] = None
    decision: str = ""
    confidence: float = 0.0
    related_issues: list[str] = Field(default_factory=list)
    reasoning: str = ""
    retrieved_count: int = 0
    candidate_count: int = 0
    top_score: Optional[float] = None
    score_gap: Optional[float] = None


class PipelineNodeInfo(BaseModel):
    name: str
    label: str
    status: NodeStatus = NodeStatus.WAITING
    elapsed_ms: Optional[int] = None
    summary: Optional[str] = None


class RunSnapshot(BaseModel):
    run_id: str
    status: RunStatus = RunStatus.QUEUED
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    total_elapsed_ms: Optional[int] = None
    current_node: Optional[str] = None
    current_round: int = 1
    issue_text: str = ""
    nodes: list[PipelineNodeInfo] = Field(default_factory=list)
    analysis: Optional[QueryAnalysisInfo] = None
    retrieval: Optional[dict[str, Any]] = None
    rerank: Optional[dict[str, Any]] = None
    decision: Optional[DecisionInfo] = None
    rounds: list[RetryRoundInfo] = Field(default_factory=list)
    node_timings_ms: dict[str, int] = Field(default_factory=dict)
    error: Optional[str] = None


class CreateRunRequest(BaseModel):
    issue_text: str = Field(min_length=1, max_length=20000, description="新 Issue 文本内容")
    sample_id: Optional[str] = Field(default=None, description="预设样例 ID（可选）")


class CreateRunResponse(BaseModel):
    run_id: str
    status: RunStatus
    created_at: str


class ExampleItem(BaseModel):
    id: str
    label: str
    tag: str
    title: str
    description: str
    expected_decision: str
    note: str


class SystemInfo(BaseModel):
    version: str
    dataset_size: int
    dataset_name: str
    embedding_model: str
    reranker_model: str
    llm_model: str
    bm25_ready: bool
    vector_ready: bool
    docstore_ready: bool
    local_demo: bool
    known_limitations: list[str]


class EvaluationSummary(BaseModel):
    sample_size: int
    total_test_queries: int
    seed: int
    self_hit_excluded: bool
    metrics: dict[str, dict[str, Any]]
    notes: list[str]
